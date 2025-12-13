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
from openai import OpenAI  # 修改：使用 OpenAI SDK
from typing import Dict, Any, List, Coroutine
from concurrent.futures import Executor

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
    if text.startswith("'''") and text.endswith("'''"):
        text = text[3:-3].strip()
    elif text.startswith('"""') and text.endswith('"""'):
        text = text[3:-3].strip()
    elif text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    elif text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and start < end:
        json_str = text[start:end + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logging.warning(f"初步JSON解析失败: {e}. 尝试修复...")
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

        # 修改：优先读取 QWEN_API_KEY，不设置代理
        self.api_key = os.environ.get("QWEN_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            logging.warning("未找到 QWEN_API_KEY 或 GEMINI_API_KEY 环境变量")

        # self.proxy = config["proxy"] # 根据要求移除代理使用
        self.dimensionality = config["embedding"]["dimensionality"]
        self.top_k = self.config["query"]["top_k"]
        self.temperature = self.config["query"]["temperature"]
        self.search_config = self.config["query"]["search_config"]

        # 修改：初始化 OpenAI 客户端 (兼容阿里云百炼)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        self.embedding_model_name = self.config["embedding"]["model"]
        self.generation_model_name = self.config["query"]["generation_model"]

        self.prompt_dir = self.config["query"]["prompt_dir"]
        db_path = PROJECT_ROOT / self.config["query"]["embedding_db_path"]
        table_name = self.config["query"]["global_table_name"]

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
        prompt_path = PROJECT_ROOT / self.prompt_dir / filename
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt文件未找到: {prompt_path}")
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _load_chunk_id_map(self) -> Dict[str, Dict[str, Any]]:
        """加载所有类型的 units 文件（text, abstract, image, table），构建 chunk ID 到源信息的映射。"""
        chunk_map = {}

        # 定义所有需要加载的 units 文件类型及其chunk类型标识
        units_files = [
            ("text_units.jsonl", "text"),
            ("abstract_units.jsonl", "abstract"),
            ("image_units.jsonl", "image"),
            ("table_units.jsonl", "table")
        ]

        chunks_dir = PROJECT_ROOT / "data" / "chunks"
        total_loaded = 0

        for filename, chunk_type in units_files:
            units_path = chunks_dir / filename

            try:
                if not units_path.exists():
                    logging.debug(f"⚠️ 未找到文件: {units_path}，跳过")
                    continue

                logging.info(f"正在加载 {filename}...")
                file_count = 0

                with open(units_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            unit = json.loads(line)
                            chunk_id = unit.get("id")
                            metadata = unit.get("metadata", {})

                            if chunk_id:
                                # 对于abstract类型，使用不同的字段
                                if chunk_type == "abstract":
                                    chunk_map[chunk_id] = {
                                        "source_filename": metadata.get("source_filename", "unknown"),
                                        "journal": metadata.get("journal", "unknown"),
                                        "year": metadata.get("year", "unknown")
                                    }
                                else:
                                    # 对于text, image, table类型，使用pages和blocks
                                    chunk_map[chunk_id] = {
                                        "source_filename": metadata.get("source_filename", "unknown"),
                                        "pages": metadata.get("pages", []),
                                        "blocks": metadata.get("blocks", [])
                                    }
                                file_count += 1

                logging.info(f"  ✓ 从 {filename} 加载了 {file_count} 个映射")
                total_loaded += file_count

            except Exception as e:
                logging.warning(f"⚠️ 加载 {filename} 时出错: {e}")
                continue

        logging.info(f"✅ 总共加载 {total_loaded} 个 chunk ID 映射关系")

        if total_loaded == 0:
            logging.warning("⚠️ 未能加载任何 chunk ID 映射，将无法解析引用")

        return chunk_map

    def _resolve_chunk_citations(self, text: str) -> str:
        """
        将文本中的 chunk ID 引用转换为来源信息。
        (逻辑保持不变，省略内部 helper 函数以节省篇幅)
        """

        def extract_base_chunk_id(full_id: str) -> str:
            match = re.match(r'(chunk-[a-f0-9]+)', full_id)
            if match: return match.group(1)
            return full_id

        def merge_pages(pages: List[int]) -> str:
            if not pages: return "Page unknown"
            pages = sorted(set(pages))
            if len(pages) == 1: return f"Page {pages[0]}"
            is_consecutive = all(pages[i + 1] - pages[i] == 1 for i in range(len(pages) - 1))
            if is_consecutive:
                return f"Page {pages[0]}-{pages[-1]}"
            else:
                return f"Pages {', '.join(map(str, pages))}"

        def merge_blocks(blocks: List[str]) -> str:
            if not blocks: return "Block unknown"
            seen = set()
            unique_blocks = []
            for b in blocks:
                if b not in seen:
                    seen.add(b)
                    unique_blocks.append(b)
            if len(unique_blocks) == 1: return f"Block {unique_blocks[0]}"
            try:
                block_nums = []
                prefix = None
                for b in unique_blocks:
                    match = re.match(r'(page_\d+_block_)(\d+)', b)
                    if match:
                        if prefix is None: prefix = match.group(1)
                        if match.group(1) == prefix:
                            block_nums.append(int(match.group(2)))
                        else:
                            block_nums = []
                            break
                if block_nums and len(block_nums) == len(unique_blocks):
                    is_consecutive = all(block_nums[i + 1] - block_nums[i] == 1 for i in range(len(block_nums) - 1))
                    if is_consecutive: return f"Block {unique_blocks[0]}~{unique_blocks[-1]}"
            except Exception:
                pass
            if len(unique_blocks) <= 3:
                return f"Blocks {', '.join(unique_blocks)}"
            else:
                return f"Blocks {unique_blocks[0]}~{unique_blocks[-1]}"

        def build_source_string(chunk_ids_raw: List[str]) -> str:
            """根据一组 chunk IDs 构建来源字符串 (English style without brackets)"""
            sources_by_file: Dict[str, Dict[str, List]] = {}
            abstract_sources = []  # 单独处理abstract类型的引用

            for chunk_id_raw in chunk_ids_raw:
                chunk_id = extract_base_chunk_id(chunk_id_raw)
                if chunk_id in self.chunk_id_map:
                    source_info = self.chunk_id_map[chunk_id]
                    chunk_type = source_info.get("chunk_type", "text")
                    filename = source_info.get("source_filename", "unknown").replace('.json', '')

                    # 对于abstract类型，使用不同的格式
                    if chunk_type == "abstract":
                        journal = source_info.get("journal", "unknown")
                        year = source_info.get("year", "unknown")
                        abstract_sources.append(f"{filename}, {journal}, {year}")
                    else:
                        # 对于text/image/table类型，使用pages和blocks
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

            source_parts = []

            # 处理abstract类型的引用
            if abstract_sources:
                source_parts.extend(abstract_sources)

            # 处理其他类型的引用
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
        """查询向量化 (修改为 OpenAI 兼容接口)"""
        logging.info(f"正在为查询进行向量化: '{query[:50]}...'")
        try:
            # 修改：使用 client.embeddings.create
            result = self.client.embeddings.create(
                model=self.embedding_model_name,
                input=query,
                dimensions=self.dimensionality  # 千问 text-embedding-v3 支持此参数
            )
            embedding_values = result.data[0].embedding

            # 手动归一化 (保留原有逻辑，text-embedding-v3 通常已归一化但双重保险)
            if self.dimensionality != 3072:
                embedding_np = np.array(embedding_values)
                norm = np.linalg.norm(embedding_np)
                if norm > 0:
                    normed_embedding = embedding_np / norm
                    logging.info(f"✅ 查询向量化并归一化成功 (Norm: {np.linalg.norm(normed_embedding):.4f})。")
                    return normed_embedding.tolist()
            else:
                logging.info("✅ 查询向量化成功 (使用默认维度，无需归一化)。")

            return embedding_values

        except Exception as e:
            logging.error(f"❌ 查询向量化失败: {e}", exc_info=True)
            raise

    def _build_context_chunks(self, query_vector: List[float]) -> list[dict] | list[Any]:
        """步骤1: 上下文构建 (逻辑不变)"""
        logging.info(f"上下文构建：正在搜索 Top {self.top_k} 相关社区...")
        try:
            results = self.community_table.search(query_vector).limit(self.top_k).to_list()
            logging.info(f"Map阶段：找到 {len(results)} 个社区。")
            return results
        except Exception as e:
            logging.error(f"❌ 在LanceDB中搜索社区失败: {e}", exc_info=True)
            return []

    def _convert_local_ids_to_chunk_ids(self, report: Dict[str, Any], id_map: Dict[str, str]) -> Dict[str, Any]:
        """将 report 中的局部 ID 转换为 chunk ID (逻辑不变)"""

        def replace_ids_in_text(text: str) -> str:
            if not isinstance(text, str): return text

            def replace_in_brackets(match):
                content = match.group(1)
                for local_id, chunk_id in id_map.items():
                    content = re.sub(r'\b' + re.escape(local_id) + r'\b', chunk_id, content)
                return f"[Data: {content}]"

            text = re.sub(r'\[Data: ([^]]+)]', replace_in_brackets, text)
            return text

        converted_report = json.loads(json.dumps(report))
        if "findings" in converted_report and isinstance(converted_report["findings"], list):
            for finding in converted_report["findings"]:
                if isinstance(finding, dict):
                    if "summary" in finding: finding["summary"] = replace_ids_in_text(finding["summary"])
                    if "explanation" in finding: finding["explanation"] = replace_ids_in_text(finding["explanation"])
        if "summary" in converted_report: converted_report["summary"] = replace_ids_in_text(converted_report["summary"])
        return converted_report

    def _generate_content_blocking(self, prompt: str) -> str:
        """
        新增：阻塞式生成函数，供异步包装器调用。
        使用 client.chat.completions.create
        """
        response = self.client.chat.completions.create(
            model=self.generation_model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature
        )
        return response.choices[0].message.content

    async def generate_async_wrapper(self, prompt: str, executor: Executor | None = None):
        """
        异步包装器 (修改为直接返回字符串内容)
        """
        loop = asyncio.get_running_loop()
        # 修改：调用新的 _generate_content_blocking 方法
        blocking_task = functools.partial(self._generate_content_blocking, prompt=prompt)
        response_content = await loop.run_in_executor(executor, blocking_task)
        return response_content

    async def _map_single_chunk(self, query: str, context_chunk: dict) -> List[Dict[str, Any]]:
        """Map阶段的单个任务"""
        try:
            payload_report_json = context_chunk.get("payload_report_json", "{}")
            payload_map_json = context_chunk.get("payload_map_json", "{}")

            report = json.loads(payload_report_json) if isinstance(payload_report_json, str) else payload_report_json
            id_map = json.loads(payload_map_json) if isinstance(payload_map_json, str) else payload_map_json

            converted_report = self._convert_local_ids_to_chunk_ids(report, id_map)

            context_data = {
                "community_id": context_chunk.get("id", "unknown"),
                "report": converted_report
            }

            context_str = json.dumps(context_data, ensure_ascii=False, indent=2)

            logging.info(f"Map阶段：处理社区 {context_data['community_id']}, 已将局部ID转换为chunk ID")

        except (json.JSONDecodeError, KeyError) as e:
            logging.warning(f"解析 payload 数据失败: {e}，使用原始数据")
            context_str = json.dumps(context_chunk, ensure_ascii=False, indent=2)

        template_prompt = Template(self.map_prompt_template)
        prompt = template_prompt.substitute(query=query, context_data=context_str)

        try:
            response_text = await self.generate_async_wrapper(prompt=prompt)

            parsed_json = try_parse_json_object(response_text)
            if parsed_json and isinstance(parsed_json.get("results"), list):
                return [
                    item for item in parsed_json["results"]
                    if "answer" in item and "score" in item
                ]
            else:
                logging.warning(f"Map阶段的LLM返回了非预期的JSON格式: {response_text[:100]}...")
                return []
        except Exception as e:
            logging.error(f"❌ Map阶段调用LLM失败: {e}", exc_info=True)
            return []

    async def _reduce_response(self, query: str, map_results: List[Dict[str, Any]]) -> str:
        """Reduce阶段"""
        logging.info(f"Reduce阶段：聚合了 {len(map_results)} 个关键点。")
        high_quality_points = [point for point in map_results if point.get("score", 0) > 0]

        if not high_quality_points:
            if not self.search_config:
                logging.warning("过滤后没有高质量的关键点，且不允许通用知识，返回无数据答案。")
                return "I am sorry but I am unable to answer your question based on the provided context."
            else:
                logging.info("无高质量关键点，但允许通用知识，将尝试直接回答。")
                report_data = "没有从知识库中找到直接相关的信息。"
        else:
            high_quality_points.sort(key=lambda x: x.get("score", 0), reverse=True)
            logging.info(f"Reduce阶段：筛选出 {len(high_quality_points)} 个高质量关键点，内容如下：{high_quality_points}")
            report_data = "\n".join(
                [f"- [得分:{point['score']}] {point['answer']}" for point in high_quality_points]
            )

        if self.search_config:
            constraints_text = "你可以利用自己的通用知识来补充和丰富回答，但必须优先使用'分析师报告'中的信息，并区分哪些信息来源于报告，哪些是你的补充知识。"
            logging.info("Reduce阶段：允许使用通用知识。")
        else:
            constraints_text = "你的回答必须严格且完全基于'分析师报告'中提供的信息。绝对不允许使用任何外部或通用知识。如果报告中的信息不足以回答问题，请明确指出这一点。"
            logging.info("Reduce阶段：严格禁止使用通用知识。")

        final_prompt = self.reduce_prompt_template.format(
            query=query,
            report_data=report_data,
            constraints=constraints_text
        )


        try:
            logging.info("Reduce阶段：调用分析师LLM生成最终答案...")
            response_text = await self.generate_async_wrapper(prompt=final_prompt)

            resolved_answer = self._resolve_chunk_citations(response_text)
            return resolved_answer
        except Exception as e:
            logging.error(f"❌ Reduce阶段调用LLM失败: {e}", exc_info=True)
            return "抱歉，在综合信息生成最终答案时发生错误。"

    async def answer_query(self, query: str) -> str:
        """执行完整的全局搜索流程"""
        try:
            query_vector = self._embed_query(query)

            context_chunks = self._build_context_chunks(query_vector)
            if not context_chunks:
                return "抱歉，我没有在知识库中找到与您的问题相关的社区信息。"

            logging.info(f"Map阶段：正在并行处理 {len(context_chunks)} 个上下文块...")
            map_tasks: List[Coroutine] = [self._map_single_chunk(query, chunk) for chunk in context_chunks]
            map_results_list = await asyncio.gather(*map_tasks)

            all_points = [point for sublist in map_results_list for point in sublist]

            final_answer = await self._reduce_response(query, all_points)
            return final_answer

        except Exception as e:
            logging.critical(f"❌ 在回答问题的过程中发生严重错误: {e}", exc_info=True)
            return "抱歉，处理您的请求时发生了一个内部错误。"


async def main():
    """主执行函数"""
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
    asyncio.run(main())