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

# core/query/basic_rag.py
import numpy as np
import argparse
import logging
import os
import textwrap
from pathlib import Path
from typing import Dict, Any, List

import lancedb
import yaml
from openai import OpenAI


# --- 项目根目录定义 ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# --- 配置日志记录 ---
def setup_logging(config: Dict[str, Any]):
    """根据配置文件设置日志记录器"""
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
    relative_log_path = log_config.get("log_file", "logs/rag_query.log")
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
    logging.info("RAG 查询日志记录器设置完成")


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


class RAGQueryHandler:
    """
    一个封装了完整 RAG (Retrieval-Augmented Generation) 流程的处理器。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化 RAG 处理器。
        - 设置 API 密钥
        - 连接到 LanceDB
        - 初始化 OpenAI 兼容客户端 (嵌入和生成)
        """
        self.config = config

        # 修改：优先读取 QWEN_API_KEY，不设置代理
        self.api_key = os.environ.get("QWEN_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            logging.warning("未找到 QWEN_API_KEY 或 GEMINI_API_KEY 环境变量")

        # self.proxy = config["proxy"] # 根据要求移除代理配置
        self.dimensionality = config["embedding"]["dimensionality"]

        # 从配置中获取参数
        db_path = PROJECT_ROOT / self.config["query"]["text_db_path"]
        prompt_path = PROJECT_ROOT / self.config["query"]["basic_rag_prompt"]
        table_name = self.config["query"]["text_table_name"]
        self.embedding_model_name = self.config["embedding"]["model"]
        self.generation_model_name = self.config["query"]["generation_model"]
        self.top_k = self.config["query"]["top_k"]
        self.temperature = self.config["query"]["temperature"]

        # --- 加载 Prompt 模板 ---
        logging.info(f"正在从 {prompt_path} 加载 Prompt 模板...")
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                self.prompt_template = f.read()
            logging.info("✅ Prompt 模板加载成功。")
        except FileNotFoundError:
            logging.critical(f"❌ Prompt 模板文件未找到: {prompt_path}", exc_info=True)
            raise

        # 初始化模型 (OpenAI 兼容)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        logging.info(f"✅ 成功初始化生成模型客户端: '{self.generation_model_name}'")

        # 连接向量数据库
        logging.info(f"正在连接到 LanceDB 数据库: {db_path}")
        try:
            db = lancedb.connect(db_path)
            self.table = db.open_table(table_name)
            logging.info(f"✅ 成功打开 LanceDB 表: '{table_name}'")
        except Exception as e:
            logging.critical(f"❌ 无法连接或打开 LanceDB 表 '{table_name}': {e}", exc_info=True)
            raise

    def embed_query(self, query_text: str) -> List[float]:
        """
        将用户的查询文本转换为向量并进行归一化处理。
        """
        logging.info(f"正在为查询进行向量化: '{query_text[:50]}...'")
        try:
            # 修改：使用 client.embeddings.create
            result = self.client.embeddings.create(
                model=self.embedding_model_name,
                input=query_text,
                dimensions=self.dimensionality  # 千问 text-embedding-v3 支持此参数
            )
            embedding_values = result.data[0].embedding

            # **CRITICAL STEP**: Normalize the embedding if not using the default 3072 dimension
            # 保留归一化逻辑，虽然千问模型通常已归一化，但保留以防万一
            if self.dimensionality != 3072:
                embedding_np = np.array(embedding_values)
                norm = np.linalg.norm(embedding_np)
                if norm > 0:
                    normed_embedding = embedding_np / norm
                    logging.info(f"✅ 查询向量化并归一化成功 (Norm: {np.linalg.norm(normed_embedding):.4f})。")
                    return normed_embedding.tolist()

            logging.info("✅ 查询向量化成功 (使用默认维度，无需归一化)。")
            return embedding_values

        except Exception as e:
            logging.error(f"❌ 查询向量化失败: {e}", exc_info=True)
            raise

    def search_vector_db(self, query_vector: List[float]) -> List[Dict]:
        """
        使用查询向量在 LanceDB 中进行相似性搜索。
        """
        logging.info(f"正在向量数据库中检索 Top-{self.top_k} 的相似文本块...")
        try:
            results = self.table.search(query_vector).limit(self.top_k).to_list()
            logging.info(f"✅ 成功检索到 {len(results)} 个结果。")
            return results
        except Exception as e:
            logging.error(f"❌ 向量检索失败: {e}", exc_info=True)
            raise

    def answer_query(self, query: str) -> str:
        """
        执行完整的 RAG 流程来回答用户的问题。
        """
        # 1. 查询向量化
        query_vector = self.embed_query(query)

        # 2. 相似性检索
        retrieved_context = self.search_vector_db(query_vector)

        # 3. 构建结构化的、带引用标识的上下文
        context_with_sources = ""
        source_map = {}  # 用于存储标识与真实来源的映射
        print("\n--- 检索到的上下文 ---\n")
        for i, doc in enumerate(retrieved_context):
            source_id = i + 1
            # 从 metadata 中提取更具体的来源信息，例如页码和块ID
            metadata = doc.get('metadata', {})
            source_filename = metadata.get('source_filename', '未知来源')
            chunk_index = metadata.get('chunk_index', '未知块')

            # 创建一个易于阅读的来源名称
            source_name = f"{source_filename} (块: {chunk_index})"
            source_map[source_id] = source_name

            # 将格式化的来源和文本追加到上下文字符串中
            context_with_sources += f"[来源 {source_id}: {source_name}]\n"
            context_with_sources += f"{doc.get('text', '')}\n\n"

            # 在控制台打印上下文，方便调试
            wrapped_text = textwrap.fill(doc.get('text', ''), width=100)
            print(f"[{source_id}] 来源: {source_name}\n{wrapped_text}\n")
            logging.info(f"[{source_id}] 来源: {source_name}\n{wrapped_text}\n")
        print("---------------------\n")

        logging.info(f"结构化上下文已构建，共 {len(retrieved_context)} 个来源。")

        # 4. 构建带有严格引用指令的 Prompt
        final_prompt = self.prompt_template.format(
            context_with_sources=context_with_sources,
            query=query
        )
        logging.info("正在调用大语言模型生成带引用的答案...")
        try:
            # 修改：使用 client.chat.completions.create
            response = self.client.chat.completions.create(
                model=self.generation_model_name,
                messages=[
                    {"role": "user", "content": final_prompt}
                ],
                temperature=self.temperature
            )
            logging.info("✅ 成功从 LLM 获取到答案。")
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"❌ 调用 LLM 生成答案时失败: {e}", exc_info=True)
            return "抱歉，在生成答案的过程中遇到了一个错误。"


def main():
    """主执行函数，用于命令行交互"""
    parser = argparse.ArgumentParser(description="使用 RAG 回答关于知识库的问题。")
    parser.add_argument("query", type=str, nargs='?', default="", help="您要提出的问题。")
    args = parser.parse_args()

    config = load_config()
    setup_logging(config)

    rag_handler = RAGQueryHandler(config)

    # 如果在命令行中提供了问题，则直接回答
    if args.query:
        logging.info(f"命令行查询: {args.query}")
        print(f"正在查询: {args.query}")
        answer = rag_handler.answer_query(args.query)
        logging.info(f"模型回答: {answer.replace(os.linesep, ' ')}")
        print("\n--- 生成的答案 ---\n")
        print(answer)
        print("\n------------------\n")
    else:
        # 否则，进入交互模式
        print("已进入交互式查询模式。输入 'exit' 或 'quit' 退出。")
        while True:
            try:
                query = input("\n请输入您的问题: ")
                if query.lower() in ["exit", "quit"]:
                    break
                logging.info(f"交互模式 - 用户输入: {query}")

                answer = rag_handler.answer_query(query)
                logging.info(f"交互模式 - 模型回答: {answer.replace(os.linesep, ' ')}")

                print("\n--- 生成的答案 ---\n")
                print(answer)
                print("\n------------------\n")
            except KeyboardInterrupt:
                print("\n再见！")
                break


if __name__ == "__main__":
    main()