import argparse
import asyncio
import json
import logging
import os
import re
import textwrap
from pathlib import Path
from string import Template
import functools
import numpy as np
import lancedb
import yaml
from google import genai
from google.genai import types
from typing import Dict, Any, List, Coroutine
from concurrent.futures import Executor
from utils.client_factory import create_gemini_client

# --- 项目根目录定义 ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- 配置日志记录 ---
def setup_logging(config: Dict[str, Any]):
    """根据配置文件设置日志记录器"""
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
    relative_log_path = log_config.get("log_file", "logs/global_query.log")
    log_file = PROJECT_ROOT / relative_log_path

    log_file.parent.mkdir(exist_ok=True, parents=True)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info("全局查询日志记录器设置完成")


# --- 加载配置 ---
def load_config(settings_filename: str = "settings.yaml") -> Dict[str, Any]:
    """加载YAML配置文件"""
    config_path = PROJECT_ROOT / "config" / settings_filename
    logging.info(f"正在从 {config_path} 加载配置...")
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件 {config_path} 未找到！")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    logging.info("配置加载成功。")
    return config


# --- JSON 解析工具 ---
def try_parse_json_object(text: str) -> Dict | None:
    """尝试解析一个可能是JSON对象的字符串，处理常见错误"""
    text = text.strip()
    # 查找第一个 '{' 和最后一个 '}'
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and start < end:
        json_str = text[start:end + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logging.warning(f"初步JSON解析失败: {e}. 尝试修复...")
            # 移除尾随逗号 (常见错误)
            json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                logging.error(f"修复后JSON解析仍然失败。原始文本: {text}")
                return None
    return None


class GlobalQueryHandler:
    """
    封装 GraphRAG 旗舰级全局搜索逻辑 (Map-Reduce)。
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.proxy = config["proxy"]
        self.dimensionality = config["embedding"]["dimensionality"]
        self.top_k = self.config["query"]["top_k"]
        self.temperature = self.config["query"]["temperature"]
        self.search_config = self.config["query"]["search_config"]

        self.client = create_gemini_client(self.api_key, self.proxy)
        self.embedding_model_name = self.config["embedding"]["model"]
        self.generation_model_name = self.config["query"]["generation_model"]

        self.prompt_dir = self.config["query"]["prompt_dir"]
        db_path=PROJECT_ROOT / self.config["query"]["embedding_db_path"]
        table_name=self.config["query"]["global_table_name"]

        self.generate = functools.partial(
            self.client.models.generate_content,
            model=f"{self.generation_model_name}",
            config=types.GenerateContentConfig(temperature=self.temperature)
        )

        try:
            self.db = lancedb.connect(db_path)
            self.community_table = self.db.open_table(table_name)
            logging.info(f"✅ 成功连接并打开LanceDB表: '{table_name}'")
        except Exception as e:
            logging.error(f"❌ 无法连接或打开LanceDB表: {e}", exc_info=True)
            raise

        self.map_prompt_template = self._load_prompt("global_map.md")
        self.reduce_prompt_template = self._load_prompt("global_reduce.md")

    def _load_prompt(self, filename: str) -> str:
        prompt_path = PROJECT_ROOT / self.prompt_dir/ filename
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt文件未找到: {prompt_path}")
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _embed_query(self, query: str) -> List[float]:
        logging.info(f"正在为查询进行向量化: '{query[:50]}...'")
        try:
            result = self.client.models.embed_content(
                model=f"{self.embedding_model_name}",
                contents=query,
                config=types.EmbedContentConfig(
                    output_dimensionality=self.dimensionality
                )
            )
            embedding_values = result.embeddings[0].values
            if self.dimensionality != 3072:
                embedding_np = np.array(embedding_values)
                normed_embedding = embedding_np / np.linalg.norm(embedding_np)
                logging.info(f"✅ 查询向量化并归一化成功 (Norm: {np.linalg.norm(normed_embedding):.4f})。")
                return normed_embedding.tolist()
            else:
                logging.info("✅ 查询向量化成功 (使用默认维度，无需归一化)。")
                return embedding_values

        except Exception as e:
            logging.error(f"❌ 查询向量化失败: {e}", exc_info=True)
            raise

    def _build_context_chunks(self, query_vector: List[float]) -> list[dict] | list[Any]:
        """
        步骤1: 上下文构建。检索相关社区摘要作为独立的文本块。
        """
        logging.info(f"上下文构建：正在搜索 Top {self.top_k} 相关社区...")
        try:
            results = self.community_table.search(query_vector).limit(self.top_k).to_list()
            logging.info(f"Map阶段：找到 {len(results)} 个社区。")
            return results
        except Exception as e:
            logging.error(f"❌ 在LanceDB中搜索社区失败: {e}", exc_info=True)
            return []

    async def generate_async_wrapper(self, prompt: str, executor: Executor | None = None):
        """
        这是一个异步包装器，它在线程池中运行同步的 generate_content 方法。
        """
        loop = asyncio.get_running_loop()
        blocking_task = functools.partial(self.generate, contents=prompt)

        # loop.run_in_executor(executor, function, *args)
        # executor: 线程池执行器。None 表示使用默认的。
        # blocking_task: 要在线程池中运行的函数。
        response = await loop.run_in_executor(executor,blocking_task)

        return response

    async def _map_single_chunk(self, query: str, context_chunk: dict) -> List[Dict[str, Any]]:
        """
        Map阶段的单个任务：从一个上下文块中提取关键点和评分。
        """
        context_str = json.dumps(context_chunk, ensure_ascii=False, indent=2)

        template_prompt= Template(self.map_prompt_template)
        prompt = template_prompt.substitute(query=query, context_data=context_str)

        try:
            response = await self.generate_async_wrapper(prompt=prompt)

            parsed_json = try_parse_json_object(response.text)
            # 确保返回的是包含 'answer' 和 'score' 的字典列表
            if parsed_json and isinstance(parsed_json.get("results"), list):
                return [
                    item for item in parsed_json["results"]
                    if "answer" in item and "score" in item
                ]
            else:
                logging.warning(f"Map阶段的LLM返回了非预期的JSON格式: {response.text}")
                return []
        except Exception as e:
            logging.error(f"❌ Map阶段调用LLM失败: {e}", exc_info=True)
            return []

    async def _reduce_response(self, query: str, map_results: List[Dict[str, Any]]) -> str:
        """
        Reduce阶段：聚合、过滤、排序，并生成最终答案。
        """
        logging.info(f"Reduce阶段：聚合了 {len(map_results)} 个关键点。")

        # 过滤掉得分为0或不相关的点
        high_quality_points = [point for point in map_results if point.get("score", 0) > 0]

        if not high_quality_points:
            if not self.search_config:
                logging.warning("过滤后没有高质量的关键点，且不允许通用知识，返回无数据答案。")
                return "I am sorry but I am unable to answer your question based on the provided context."
            logging.info("无高质量关键点，但允许通用知识，将尝试直接回答。")
            report_data = "没有从知识库中找到直接相关的信息。"
        else:
            # 按得分降序排序
            high_quality_points.sort(key=lambda x: x.get("score", 0), reverse=True)
            logging.info(f"Reduce阶段：筛选出 {len(high_quality_points)} 个高质量关键点，内容如下：{high_quality_points}")

            # 格式化成“分析师报告”
            report_data = "\n".join(
                [f"- [得分:{point['score']}] {point['answer']}" for point in high_quality_points]
            )

        # 根据配置动态生成约束指令
        if self.search_config:
            constraints_text = "你可以利用自己的通用知识来补充和丰富回答，但必须优先使用'分析师报告'中的信息，并区分哪些信息来源于报告，哪些是你的补充知识。"
            logging.info("Reduce阶段：允许使用通用知识。")
        else:
            constraints_text = "你的回答必须严格且完全基于'分析师报告'中提供的信息。绝对不允许使用任何外部或通用知识。如果报告中的信息不足以回答问题，请明确指出这一点。"
            logging.info("Reduce阶段：严格禁止使用通用知识。")

        final_prompt = self.reduce_prompt_template.format(query=query, report_data=report_data, constraints=constraints_text)

        try:
            logging.info("Reduce阶段：调用分析师LLM生成最终答案...")
            response = await self.generate_async_wrapper(prompt=final_prompt)

            return response.text
        except Exception as e:
            logging.error(f"❌ Reduce阶段调用LLM失败: {e}", exc_info=True)
            return "抱歉，在综合信息生成最终答案时发生错误。"

    async def answer_query(self, query: str) -> str:
        """
        执行完整的、符合GraphRAG旗舰实现的全局搜索流程。
        """
        try:
            # 1. 向量化查询
            query_vector = self._embed_query(query)

            # 2. 上下文构建
            context_chunks = self._build_context_chunks(query_vector)
            if not context_chunks:
                return "抱歉，我没有在知识库中找到与您的问题相关的社区信息。"

            # 3. 并行Map阶段
            logging.info(f"Map阶段：正在并行处理 {len(context_chunks)} 个上下文块...")
            map_tasks: List[Coroutine] = [self._map_single_chunk(query, chunk) for chunk in context_chunks]
            map_results_list = await asyncio.gather(*map_tasks)

            # 展开所有结果
            all_points = [point for sublist in map_results_list for point in sublist]

            # 4. Reduce阶段
            final_answer = await self._reduce_response(query, all_points)
            return final_answer

        except Exception as e:
            logging.critical(f"❌ 在回答问题的过程中发生严重错误: {e}", exc_info=True)
            return "抱歉，处理您的请求时发生了一个内部错误。"


async def main():
    """主执行函数，用于命令行交互"""
    parser = argparse.ArgumentParser(description="使用GraphRAG旗舰级全局搜索(Map-Reduce)回答问题。")
    parser.add_argument("query", type=str, nargs='?', default="", help="您要提出的问题。")
    args = parser.parse_args()

    try:
        config = load_config()
        setup_logging(config)
        handler = GlobalQueryHandler(config)

        if args.query:
            print(f"正在查询: {args.query}")
            answer = await handler.answer_query(args.query)
            print("\n--- 全局搜索生成的答案 ---\n")
            print(answer)
            print("\n--------------------------\n")
        else:
            print("已进入交互式查询模式。输入 'exit' 或 'quit' 退出。")
            while True:
                try:
                    query = input("\n请输入您的问题: ")
                    if query.lower() in ["exit", "quit"]:
                        break
                    answer = await handler.answer_query(query)
                    print("\n--- 全局搜索生成的答案 ---\n")
                    print(answer)
                    logging.info(f"全局搜索生成的答案: {answer}")
                    print("\n--------------------------\n")
                except (KeyboardInterrupt, EOFError):
                    print("\n再见！")
                    break

    except (FileNotFoundError, Exception) as e:
        logging.critical(f"程序启动或运行失败: {e}", exc_info=True)
        print(f"发生严重错误，请查看日志文件 `logs/global_query.log`。错误: {e}")


if __name__ == "__main__":
    # 使用 asyncio.run 来执行 async main()
    asyncio.run(main())