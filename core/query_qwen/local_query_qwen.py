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
from typing import Dict, Any, List
from concurrent.futures import Executor
from utils.local_context import LocalSearchContextBuilder

# --- 项目根目录定义 ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# --- 配置日志记录 ---
def setup_logging(config: Dict[str, Any]):
    """根据配置文件设置日志记录器"""
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
    relative_log_path = log_config.get("log_file", "logs/local_query.log")
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
    logging.info("局部查询日志记录器设置完成")


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


class LocalQueryHandler:
    """
    思路：Query Embedding -> Entity Search -> Context Building -> LLM Generation
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # 修改：优先读取阿里云 API Key，兼容旧环境变量
        self.api_key = os.getenv("QWEN_API_KEY")
        if not self.api_key:
            logging.warning("未找到 QWEN_API_KEY 或 GEMINI_API_KEY 环境变量")

        self.dimensionality = config["embedding"]["dimensionality"]

        # 局部查询特定的配置参数
        self.top_k = self.config["query"].get("local_top_k", 10)
        self.temperature = self.config["query"]["temperature"]
        self.search_config = self.config["query"]["search_config"]

        # Token限制
        self.max_context_tokens = self.config["query"].get("max_local_context_tokens", 12000)

        # K-hop expansion depth for neighborhood
        self.k_hop = self.config["query"].get("k_hop", 1)

        http_client_kwargs = {}

        # 修改：初始化 OpenAI 客户端 (兼容阿里云百炼)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        **http_client_kwargs
        )

        self.embedding_model_name = self.config["embedding"]["model"]
        self.generation_model_name = self.config["query"]["generation_model"]

        self.prompt_dir = self.config["query"]["prompt_dir"]
        db_path = PROJECT_ROOT / self.config["query"]["embedding_db_path"]

        table_name = self.config["query"].get("local_table_name", "entities")

        try:
            self.db = lancedb.connect(db_path)
            self.entity_table = self.db.open_table(table_name)
            logging.info(f"✅ 成功连接并打开LanceDB实体表: '{table_name}'")
        except Exception as e:
            logging.error(f"❌ 无法连接或打开LanceDB表 '{table_name}': {e}", exc_info=True)
            raise

        # 加载提示词模板
        self.local_prompt_template = self._load_prompt("local_query.md")

        # 加载 chunk ID 到源信息的映射
        self.chunk_id_map = self._load_chunk_id_map()

        # 加载图数据、文本单元和社区报告用于上下文构建
        self.graph_data = self._load_graph_data()
        self.text_units = self._load_text_units()
        self.community_reports = self._load_community_reports()

        # 初始化局部上下文构建器
        self.context_builder = LocalSearchContextBuilder(
            graph_data=self.graph_data,
            text_units=self.text_units,
            community_reports=self.community_reports,
            context_token_limit=self.max_context_tokens
        )

    def _load_prompt(self, filename: str) -> str:
        prompt_path = PROJECT_ROOT / self.prompt_dir / filename
        if not prompt_path.exists():
            logging.warning(f"⚠️ Prompt文件未找到: {prompt_path}，使用内置默认模板。")
            return """
            请根据以下提供的上下文信息回答用户的问题。
            上下文信息 (包含实体描述和关联的数据来源ID):
            ${context_data}
            用户问题: ${query}
            ${constraints}
            请在回答中引用来源。
            """
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _load_chunk_id_map(self) -> Dict[str, Dict[str, Any]]:
        """加载 text_units.jsonl 文件，构建 chunk ID 到源信息的映射。"""
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

    def _load_graph_data(self) -> Dict[str, Any]:
        """加载 final_graph.json 数据"""
        graph_path = PROJECT_ROOT / self.config["query"]["input_graph_path"]
        try:
            logging.info(f"正在加载图数据: {graph_path}")
            with open(graph_path, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            logging.info(f"✅ 成功加载图数据，包含 {len(graph_data.get('nodes', []))} 个节点")
            return graph_data
        except FileNotFoundError:
            logging.warning(f"⚠️ 未找到图数据文件: {graph_path}")
            return {"nodes": [], "links": []}
        except Exception as e:
            logging.error(f"❌ 加载图数据时出错: {e}", exc_info=True)
            return {"nodes": [], "links": []}

    def _load_text_units(self) -> Dict[str, str]:
        """加载文本单元，构建 chunk_id -> text 的映射"""
        text_units_path = PROJECT_ROOT / self.config["embedding"]["input_text_units_path"]
        text_map = {}
        try:
            logging.info(f"正在加载文本单元: {text_units_path}")
            with open(text_units_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        unit = json.loads(line)
                        chunk_id = unit.get("id")
                        text = unit.get("text", "")
                        if chunk_id and text:
                            text_map[chunk_id] = text
            logging.info(f"✅ 成功加载 {len(text_map)} 个文本单元")
            return text_map
        except FileNotFoundError:
            logging.warning(f"⚠️ 未找到文本单元文件: {text_units_path}")
            return {}
        except Exception as e:
            logging.error(f"❌ 加载文本单元时出错: {e}", exc_info=True)
            return {}

    def _load_community_reports(self) -> Dict[str, str]:
        """加载社区报告，构建 community_id -> report 的映射"""
        reports_path = PROJECT_ROOT / self.config["query"]["input_community_report_path"]
        reports_map = {}
        try:
            logging.info(f"正在加载社区报告: {reports_path}")
            with open(reports_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        report_data = json.loads(line)
                        community_id = report_data.get("community_id")
                        report = report_data.get("report", {})
                        if community_id and report:
                            report_str = f"**{report.get('title', 'Community Report')}**\n\n"
                            report_str += f"{report.get('summary', '')}\n\n"
                            findings = report.get("findings", [])
                            if findings:
                                report_str += "**Key Findings:**\n"
                                for finding in findings:
                                    report_str += f"- {finding.get('summary', '')}\n"
                            reports_map[community_id] = report_str
            logging.info(f"✅ 成功加载 {len(reports_map)} 个社区报告")
            return reports_map
        except FileNotFoundError:
            logging.warning(f"⚠️ 未找到社区报告文件: {reports_path}")
            return {}
        except Exception as e:
            logging.error(f"❌ 加载社区报告时出错: {e}", exc_info=True)
            return {}

    def _resolve_chunk_citations(self, text: str) -> str:
        """
        将文本中的 chunk ID 引用转换为来源信息。
        (保持原有逻辑不变，省略内部 helper 函数以节省篇幅，功能完全一致)
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
                    if "unknown" not in sources_by_file:
                        sources_by_file["unknown"] = {"pages": [], "blocks": []}
            if not sources_by_file: return ""
            source_parts = []
            for filename, info in sources_by_file.items():
                if filename == "unknown": continue
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
        logging.info(f"正在为查询进行向量化: '{query[:2000]}...'")
        try:
            # 修改：使用 client.embeddings.create
            # 注意：需确保 config 中的 model 是兼容的（如 text-embedding-v3）
            result = self.client.embeddings.create(
                model=self.embedding_model_name,
                input=query,
                dimensions=self.dimensionality  # 千问 text-embedding-v3 支持此参数
            )
            # 获取嵌入向量
            embedding_values = result.data[0].embedding

            # 手动归一化 (虽然 text-embedding-v3 通常已归一化，但为了保险)
            embedding_np = np.array(embedding_values)
            norm = np.linalg.norm(embedding_np)
            if norm > 0:
                normed_embedding = embedding_np / norm
                return normed_embedding.tolist()
            return embedding_values

        except Exception as e:
            logging.error(f"❌ 查询向量化失败: {e}", exc_info=True)
            raise

    def _build_local_context(self, query_vector: List[float]) -> str:
        """构建局部查询上下文 (逻辑不变)"""
        logging.info(f"正在搜索 Top {self.top_k} 相关实体...")
        try:
            results = self.entity_table.search(query_vector).limit(self.top_k).to_list()
            if not results:
                logging.warning("向量检索未找到任何相关实体")
                return ""
            logging.info(f"✅ 检索到 {len(results)} 个相关实体")

            context = self.context_builder.build(
                selected_entities=results,
                k_hop=self.k_hop
            )
            if not context:
                logging.warning("LocalSearchContextBuilder 未能构建有效上下文")
                return ""
            logging.info(f"✅ 成功构建局部上下文，长度约 {len(context)} 字符")
            return context
        except Exception as e:
            logging.error(f"❌ 构建局部上下文失败: {e}", exc_info=True)
            return ""

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
        """异步包装器 (修改为直接返回字符串内容)"""
        loop = asyncio.get_running_loop()
        # 修改：调用新的 _generate_content_blocking 方法
        blocking_task = functools.partial(self._generate_content_blocking, prompt=prompt)
        response_content = await loop.run_in_executor(executor, blocking_task)
        return response_content

    async def answer_query(self, query: str) -> str:
        """执行局部查询流程"""
        try:
            # 1. 向量化查询
            query_vector = self._embed_query(query)

            # 2. 构建局部上下文
            context_data = self._build_local_context(query_vector)

            if not context_data:
                if not self.search_config:
                    return "抱歉，我没有在知识库中找到与您的问题相关的实体信息。"
                else:
                    logging.info("未找到相关实体，但允许通用知识，将仅基于通用知识回答。")
                    context_data = "知识库中没有检索到直接相关的实体。"

            # 3. 准备 Prompt
            if self.search_config:
                constraints_text = "你可以利用自己的通用知识来补充和丰富回答，但必须优先使用提供的'上下文信息'。请明确区分哪些信息来源于上下文，哪些是你的补充知识。"
            else:
                constraints_text = "你的回答必须严格且完全基于'上下文信息'中提供的内容。绝对不允许使用任何外部或通用知识。"

            template = Template(self.local_prompt_template)
            prompt = template.safe_substitute(
                context_data=context_data,
                query=query,
                constraints=constraints_text
            )

            # 4. 生成回答
            logging.info("调用LLM生成局部查询答案...")
            # 修改：此处直接获取内容字符串，不再是对象
            response_text = await self.generate_async_wrapper(prompt=prompt)

            # 5. 解析引用
            resolved_answer = self._resolve_chunk_citations(response_text)
            return resolved_answer

        except Exception as e:
            logging.critical(f"❌ 在局部查询过程中发生严重错误: {e}", exc_info=True)
            return "抱歉，处理您的局部查询请求时发生了一个内部错误。"


async def main():
    """主执行函数"""
    parser = argparse.ArgumentParser(description="使用局部搜索(Local Search)回答问题。")
    parser.add_argument("query", type=str, nargs='?', default="", help="您要提出的问题。")
    args = parser.parse_args()

    try:
        config = load_config()
        setup_logging(config)
        handler = LocalQueryHandler(config)

        if args.query:
            print(f"正在查询 (Local): {args.query}")
            answer = await handler.answer_query(args.query)
            print("\n--- 局部搜索生成的答案 ---\n")
            print(answer)
            logging.info(f"局部搜索生成的答案: {answer}")
            print("\n--------------------------\n")
        else:
            print("已进入局部查询交互模式。输入 'exit' 或 'quit' 退出。")
            while True:
                try:
                    query = input("\n请输入您的问题: ")
                    if query.lower() in ["exit", "quit"]:
                        break
                    answer = await handler.answer_query(query)
                    print("\n--- 局部搜索生成的答案 ---\n")
                    print(answer)
                    logging.info(f"局部搜索生成的答案: {answer}")
                    print("\n--------------------------\n")
                except (KeyboardInterrupt, EOFError):
                    print("\n再见！")
                    break

    except (FileNotFoundError, Exception) as e:
        logging.critical(f"程序启动或运行失败: {e}", exc_info=True)
        print(f"发生严重错误，请查看日志文件。错误: {e}")


if __name__ == "__main__":
    asyncio.run(main())