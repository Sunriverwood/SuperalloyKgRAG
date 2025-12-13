# Copyright 2025 SUNRIVERWOOD
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import asyncio
import json
import logging
import os
import re
from pathlib import Path
from string import Template
import functools
import numpy as np
import lancedb
import yaml
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
    # 移除可能的三重引号包裹
    if text.startswith("'''") and text.endswith("'''"):
        text = text[3:-3].strip()
    elif text.startswith('"""') and text.endswith('"""'):
        text = text[3:-3].strip()
    elif text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    elif text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
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

        # 加载 chunk ID 到源信息的映射
        self.chunk_id_map = self._load_chunk_id_map()

    def _load_prompt(self, filename: str) -> str:
        prompt_path = PROJECT_ROOT / self.prompt_dir/ filename
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt文件未找到: {prompt_path}")
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _load_chunk_id_map(self) -> Dict[str, Dict[str, Any]]:
        """
        加载 text_units.jsonl 文件，构建 chunk ID 到源信息的映射。
        返回格式: {chunk_id: {"source_filename": str, "pages": list, "blocks": list}}
        """
        text_units_path = PROJECT_ROOT / self.config["embedding"]["input_text_units_path"]
        chunk_map = {}

        try:
            logging.info(f"正在加载文本单元映射: {text_units_path}")
            with open(text_units_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        unit = json.loads(line)
                        chunk_id = unit.get("id")
                        metadata = unit.get("metadata", {})

                        if chunk_id:
                            chunk_map[chunk_id] = {
                                "source_filename": metadata.get("source_filename", "unknown"),
                                "pages": metadata.get("pages", []),
                                "blocks": metadata.get("blocks", [])
                            }

            logging.info(f"✅ 成功加载 {len(chunk_map)} 个文本单元的映射关系")
            return chunk_map

        except FileNotFoundError:
            logging.warning(f"⚠️ 未找到文本单元文件: {text_units_path}，将无法解析chunk ID引用")
            return {}
        except Exception as e:
            logging.error(f"❌ 加载文本单元映射时出错: {e}", exc_info=True)
            return {}

    def _resolve_chunk_citations(self, text: str) -> str:
        """
        将文本中的 chunk ID 引用转换为来源信息。
        支持以下格式：
        1. [Data: Entities (chunk-xxx, chunk-yyy)] → [来源: 文件名, Page 1-3; Block page_1_block_1~page_1_block_3]
        2. [cite: chunk-xxx, chunk-yyy] → 同样转换为 [来源: ...]
        并移除正文中出现的局部 ID 标记，例如：（E1）、(E2)、（R3）等。

        支持来源合并：同一句中同一文件的多个chunk会合并页码范围和区块范围。
        """
        def extract_base_chunk_id(full_id: str) -> str:
            """提取基础的 chunk ID，去除 -e-X 或 -r-X 等后缀"""
            match = re.match(r'(chunk-[a-f0-9]+)', full_id)
            if match:
                return match.group(1)
            return full_id

        def merge_pages(pages: List[int]) -> str:
            """合并页码为范围（相邻）或列表（不相邻）"""
            if not pages:
                return "Page unknown"
            pages = sorted(set(pages))
            if len(pages) == 1:
                return f"Page {pages[0]}"
            is_consecutive = all(pages[i+1] - pages[i] == 1 for i in range(len(pages)-1))
            if is_consecutive:
                return f"Page {pages[0]}-{pages[-1]}"
            else:
                return f"Pages {', '.join(map(str, pages))}"

        def merge_blocks(blocks: List[str]) -> str:
            """合并区块为范围（相邻）或列表（不相邻）"""
            if not blocks:
                return "Block unknown"
            seen = set()
            unique_blocks = []
            for b in blocks:
                if b not in seen:
                    seen.add(b)
                    unique_blocks.append(b)
            if len(unique_blocks) == 1:
                return f"Block {unique_blocks[0]}"
            try:
                block_nums = []
                prefix = None
                for b in unique_blocks:
                    match = re.match(r'(page_\d+_block_)(\d+)', b)
                    if match:
                        if prefix is None:
                            prefix = match.group(1)
                        if match.group(1) == prefix:
                            block_nums.append(int(match.group(2)))
                        else:
                            block_nums = []
                            break
                if block_nums and len(block_nums) == len(unique_blocks):
                    is_consecutive = all(block_nums[i+1] - block_nums[i] == 1 for i in range(len(block_nums)-1))
                    if is_consecutive:
                        return f"Block {unique_blocks[0]}~{unique_blocks[-1]}"
            except Exception:
                pass
            if len(unique_blocks) <= 3:
                return f"Blocks {', '.join(unique_blocks)}"
            else:
                return f"Blocks {unique_blocks[0]}~{unique_blocks[-1]}"

        def build_source_string(chunk_ids_raw: List[str]) -> str:
            """根据一组 chunk IDs 构建来源字符串 (English style without brackets)"""
            sources_by_file: Dict[str, Dict[str, List]] = {}
            for chunk_id_raw in chunk_ids_raw:
                chunk_id = extract_base_chunk_id(chunk_id_raw)
                if chunk_id in self.chunk_id_map:
                    source_info = self.chunk_id_map[chunk_id]
                    filename = source_info.get("source_filename", "unknown").replace('.json', '')
                    pages = source_info.get("pages", [])
                    blocks = source_info.get("blocks", [])
                    if filename not in sources_by_file:
                        sources_by_file[filename] = {"pages": [], "blocks": []}
                    sources_by_file[filename]["pages"].extend(pages)
                    sources_by_file[filename]["blocks"].extend(blocks)
                else:
                    logging.warning(f"⚠️ 未找到 chunk ID 的映射: {chunk_id} (原始ID: {chunk_id_raw})")
                    if "unknown" not in sources_by_file:
                        sources_by_file["unknown"] = {"pages": [], "blocks": []}
            if not sources_by_file:
                return "[source: unknown]"
            source_parts = []
            for filename, info in sources_by_file.items():
                if filename == "unknown":
                    continue
                page_str = merge_pages(info["pages"])
                block_str = merge_blocks(info["blocks"])
                source_parts.append(f"{filename} {page_str} {block_str}")
            if len(source_parts) == 0:
                return "[source: unknown]"
            if len(source_parts) == 1:
                return f"[source: {source_parts[0]}]"
            return f"[source: {'; '.join(source_parts)}]"

        def replace_data_citation(match):
            # prefix = match.group(1).strip()
            chunk_ids_raw = match.group(2)
            ids_raw = [cid.strip() for cid in chunk_ids_raw.split(',') if cid.strip()]
            return build_source_string(ids_raw)

        def replace_cite(match):
            chunk_ids_raw = match.group(1)
            ids_raw = [cid.strip() for cid in chunk_ids_raw.split(',') if cid.strip()]
            return build_source_string(ids_raw)

        # 1. 处理 [Data: Entities (...)] / [Data: Relationships (...)] 等格式
        text = re.sub(r'\[Data:\s*([^(]+)\(([^)]+)\)]', replace_data_citation, text)
        # 2. 处理 [cite: chunk-xxx, chunk-yyy] 格式
        text = re.sub(r'\[cite:\s*([^]]+)]', replace_cite, text)
        # 3. 去除正文中的局部 ID 标记，例如 （E1）、(E2)、（R3） 等
        text = re.sub(r'[（(][ER]\d+[）)]', '', text)
        # 4. 去除可能产生的多余双空格
        text = re.sub(r'\s{2,}', ' ', text)
        return text

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
        同时获取 payload_report_json 和 payload_map_json。
        """
        logging.info(f"上下文构建：正在搜索 Top {self.top_k} 相关社区...")
        try:
            results = self.community_table.search(query_vector).limit(self.top_k).to_list()
            logging.info(f"Map阶段：找到 {len(results)} 个社区。")
            return results
        except Exception as e:
            logging.error(f"❌ 在LanceDB中搜索社区失败: {e}", exc_info=True)
            return []

    def _convert_local_ids_to_chunk_ids(self, report: Dict[str, Any], id_map: Dict[str, str]) -> Dict[str, Any]:
        """
        将 report 中的局部 ID (E1, E2, R1, R2 等) 转换为实际的 chunk ID。

        Args:
            report: 包含局部 ID 的社区报告
            id_map: 局部 ID 到 chunk ID 的映射关系

        Returns:
            转换后的报告，其中所有局部 ID 都替换为对应的 chunk ID
        """

        def replace_ids_in_text(text: str) -> str:
            """在文本中替换所有的局部 ID 引用"""
            if not isinstance(text, str):
                return text

            # 匹配 [Data: Entities (E1, E2, ...); Relationships (R1, R2, ...)] 格式
            def replace_in_brackets(match):
                content = match.group(1)
                # 替换所有 E 和 R 开头的 ID
                for local_id, chunk_id in id_map.items():
                    content = re.sub(r'\b' + re.escape(local_id) + r'\b', chunk_id, content)
                return f"[Data: {content}]"

            # 替换 Data 引用中的 ID
            text = re.sub(r'\[Data: ([^]]+)]', replace_in_brackets, text)

            return text

        # 深拷贝报告以避免修改原始数据
        converted_report = json.loads(json.dumps(report))

        # 转换 findings 中的所有文本
        if "findings" in converted_report and isinstance(converted_report["findings"], list):
            for finding in converted_report["findings"]:
                if isinstance(finding, dict):
                    if "summary" in finding:
                        finding["summary"] = replace_ids_in_text(finding["summary"])
                    if "explanation" in finding:
                        finding["explanation"] = replace_ids_in_text(finding["explanation"])

        # 转换 summary
        if "summary" in converted_report:
            converted_report["summary"] = replace_ids_in_text(converted_report["summary"])

        return converted_report

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
        使用 payload_report_json 和 payload_map_json 来重构带有真实 chunk ID 的文本。
        """
        try:
            # 提取 payload 数据
            payload_report_json = context_chunk.get("payload_report_json", "{}")
            payload_map_json = context_chunk.get("payload_map_json", "{}")

            # 解析 JSON
            report = json.loads(payload_report_json) if isinstance(payload_report_json, str) else payload_report_json
            id_map = json.loads(payload_map_json) if isinstance(payload_map_json, str) else payload_map_json

            # 转换局部 ID 为 chunk ID
            converted_report = self._convert_local_ids_to_chunk_ids(report, id_map)

            # 构建用于 LLM 的上下文，使用转换后的报告
            context_data = {
                "community_id": context_chunk.get("id", "unknown"),
                "report": converted_report
            }

            context_str = json.dumps(context_data, ensure_ascii=False, indent=2)

            logging.info(f"Map阶段：处理社区 {context_data['community_id']}, 已将局部ID转换为chunk ID")

        except (json.JSONDecodeError, KeyError) as e:
            logging.warning(f"解析 payload 数据失败: {e}，使用原始数据")
            # 如果解析失败，使用原始数据
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
            else:
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

            # 将 chunk ID 引用转换为可读的源信息
            resolved_answer = self._resolve_chunk_citations(response.text)
            return resolved_answer
        except Exception as e:
            logging.error(f"❌ Reduce阶段调用LLM失败: {e}", exc_info=True)
            return "抱歉，在综合信息生成最终答案时发生错误。"

    async def answer_query(self, query: str) -> str:
        """
        执行完整的全局搜索流程。
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
    parser = argparse.ArgumentParser(description="使用全局搜索(Map-Reduce)回答问题。")
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