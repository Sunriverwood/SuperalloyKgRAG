# M3GraphRAG/core/query/global_query.py
import argparse
import logging
import os
import textwrap
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import lancedb
import yaml
from google.genai import types

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
    # 避免重复添加handler
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
def load_config(settings_path: Path = PROJECT_ROOT / "config/settings.yaml") -> Dict[str, Any]:
    """加载YAML配置文件"""
    if not settings_path.is_file():
        raise FileNotFoundError(f"配置文件未找到: {settings_path}")
    with open(settings_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class GlobalQueryHandler:
    """
    封装全局搜索的完整逻辑 (Map-Reduce)。
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.proxy = config["proxy"]
        self.dimensionality = config["embedding"]["dimensionality"]
        self.top_k = self.config["query"]["top_k"]
        self.temperature= self.config["query"]["temperature"]
        # --- 初始化客户端和模型 ---
        self.client = create_gemini_client(self.api_key, self.proxy)
        self.generation_model_name = self.config["query"]["generation_model"]
        self.embedding_model_name = self.config["embedding"]["model"]

        logging.info(f"LLM模型: {self.generation_model_name}, Embedding模型: {self.embedding_model_name}")

        table_name = self.config["query"]["global_table_name"]

        # --- 连接 LanceDB ---
        db_path = PROJECT_ROOT / self.config["query"]["embedding_db_path"]
        try:
            self.db = lancedb.connect(db_path)
            logging.info(f"成功连接到 LanceDB at {db_path}")
            self.community_table = self.db.open_table(table_name)
            logging.info(f"成功打开社区表: '{table_name}'")
        except Exception as e:
            logging.error(f"❌ 无法连接或打开LanceDB表: {e}", exc_info=True)
            raise

        # --- 加载 Prompts ---
        self.reduce_prompt_template = self._load_prompt("global_retrieval.md")
        self.answer_prompt_template = self._load_prompt("global_generation.md")

    def _load_prompt(self, filename: str) -> str:
        """从config/prompts加载prompt模板"""
        prompt_path = PROJECT_ROOT / self.config["query"]["prompt_dir"] / filename
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt文件未找到: {prompt_path}")
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _embed_query(self, query: str) -> List[float]:
        """将用户查询文本向量化"""
        try:
            logging.info(f"正在向量化查询...")
            result = self.client.models.embed_content(
                model=f"{self.embedding_model_name}",
                contents=query,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",  # Correct task type for queries [cite: 81]
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

    def _map_communities(self, query_vector: List[float]) -> List[Dict[str, Any]]:
        """
        Map阶段：在LanceDB中搜索最相关的社区。
        """
        logging.info(f"Map阶段：正在搜索 Top {self.top_k} 相关社区...")
        try:
            results = self.community_table.search(query_vector).limit(self.top_k).to_list()
            logging.info(f"Map阶段：找到 {len(results)} 个社区。")
            return results
        except Exception as e:
            logging.error(f"❌ 在LanceDB中搜索社区失败: {e}", exc_info=True)
            return []

    def _reduce_context(self, query: str, communities: List[Dict[str, Any]]) -> str:
        """
        Reduce阶段：使用LLM将社区摘要精炼成统一的上下文。
        """
        if not communities:
            return ""

        logging.info("Reduce阶段：正在综合社区摘要...")

        # 将社区摘要格式化
        context_to_reduce = ""
        for i, comm in enumerate(communities):
            context_to_reduce += f"--- 相关社区 {i + 1} ---\n摘要: {comm.get('text', 'N/A')}\n\n"

        prompt = self.reduce_prompt_template.format(
            query=query,
            context=context_to_reduce
        )

        try:
            response = self._invoke_llm(prompt)
            logging.info("Reduce阶段：上下文综合完成。")
            return response
        except Exception as e:
            logging.error(f"❌ Reduce阶段调用LLM失败: {e}", exc_info=True)
            # 如果精炼失败，退回到简单的拼接作为上下文
            return context_to_reduce

    def _invoke_llm(self, prompt: str) -> str:
        """封装调用LLM的逻辑"""
        try:
            response = self.client.models.generate_content(
                model=f"{self.generation_model_name}",
                contents=prompt,
                config=types.GenerateContentConfig(temperature=self.temperature)
            )
            return response.text
        except Exception as e:
            logging.error(f"❌ 调用LLM失败: {e}", exc_info=True)
            raise

    def answer_query(self, query: str) -> str:
        """
        执行完整的全局搜索RAG流程。
        """
        try:
            # 1. 向量化查询
            query_vector = self._embed_query(query)

            # 2. Map阶段：查找相关社区
            top_communities = self._map_communities(query_vector)
            if not top_communities:
                return "抱歉，我没有在知识库中找到与您的问题相关的社区信息。"

            # 3. Reduce阶段：精炼上下文
            reduced_context = self._reduce_context(query, top_communities)
            logging.info(f"--- 精炼后的上下文 ---\n{textwrap.shorten(reduced_context, 200)}\n--------------------")

            # 4. 构建最终Prompt
            final_prompt = self.answer_prompt_template.format(
                query=query,
                context=reduced_context
            )

            # 5. 调用LLM生成最终答案
            logging.info("正在基于精炼后的上下文生成最终答案...")
            final_answer = self._invoke_llm(final_prompt)
            return final_answer

        except Exception as e:
            logging.error(f"❌ 在回答问题的过程中发生严重错误: {e}", exc_info=True)
            return "抱歉，处理您的请求时发生了一个内部错误。"


def main():
    """主执行函数，用于命令行交互"""
    parser = argparse.ArgumentParser(description="使用全局搜索(Map-Reduce)RAG回答关于知识库的问题。")
    parser.add_argument("query", type=str, nargs='?', default="", help="您要提出的问题。")
    args = parser.parse_args()

    try:
        config = load_config()
        setup_logging(config)
        handler = GlobalQueryHandler(config)

        if args.query:
            logging.info(f"命令行查询: {args.query}")
            print(f"正在查询: {args.query}")
            answer = handler.answer_query(args.query)
            logging.info(f"模型回答: {answer.replace(os.linesep, ' ')}")
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
                    logging.info(f"交互式查询: {query}")
                    answer = handler.answer_query(query)
                    logging.info(f"模型回答: {answer.replace(os.linesep, ' ')}")
                    print("\n--- 全局搜索生成的答案 ---\n")
                    print(answer)
                    print("\n--------------------------\n")
                except KeyboardInterrupt:
                    print("\n再见！")
                    break

    except FileNotFoundError as e:
        logging.error(e)
        print(f"错误: {e}")
    except Exception as e:
        logging.critical(f"程序启动失败: {e}", exc_info=True)
        print(f"发生严重错误，请查看日志文件。")


if __name__ == "__main__":
    main()