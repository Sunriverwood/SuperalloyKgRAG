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

import asyncio
import json
import logging
import re
from typing import Dict, Any, List, Literal
from string import Template
import yaml
from pathlib import Path
import os

# 复用现有模块
from global_query import GlobalQueryHandler
from local_query import LocalQueryHandler
from utils.client_factory import create_gemini_client

# --- 项目根目录定义 ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(settings_filename: str = "settings.yaml") -> Dict[str, Any]:
    """复用配置加载逻辑"""
    config_path = PROJECT_ROOT / "config" / settings_filename
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件 {config_path} 未找到！")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_logging(config: Dict[str, Any]):
    """设置路由专用日志"""
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
    relative_log_path = log_config.get("log_file", "logs/router.log")  # 修改日志文件名
    log_file = PROJECT_ROOT / relative_log_path
    log_file.parent.mkdir(exist_ok=True, parents=True)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - [Router] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


class DriftSearchHandler(LocalQueryHandler):
    """
    漂移搜索处理器 (Drift Search Handler)
    继承自 LocalQueryHandler，增加了多轮检索和上下文扩展能力。
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.drift_k_followups = config["query"].get("drift_k_followups", 2)  # 每次漂移生成的后续问题数
        self.drift_max_steps = config["query"].get("drift_max_steps", 2)  # 最大漂移轮数

    async def _generate_follow_up_queries(self, original_query: str, current_context: str) -> List[str]:
        """
        反思阶段：基于当前上下文，判断是否需要更多信息，并生成后续查询。
        """
        prompt = f"""
        你是一个智能搜索助手。用户的问题是: "{original_query}"

        目前检索到的上下文信息如下:
        {current_context[:3000]}... (content truncated)

        请评估上述上下文是否足以完全回答用户的问题。
        - 如果足以回答，请输出空列表 []。
        - 如果不足以回答，请生成 1 到 {self.drift_k_followups} 个简短的后续搜索关键词或问题，用于在知识图谱中检索缺失的信息。

        请仅返回一个 JSON 格式的字符串列表，例如: ["相关实体A", "实体B的属性"]
        """

        try:
            response = await self.generate_async_wrapper(prompt=prompt)
            text = response.text.strip()
            # 简单的 JSON 提取
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return []
        except Exception as e:
            logging.warning(f"生成后续查询失败: {e}")
            return []

    async def perform_drift_search(self, query: str) -> str:
        """
        执行漂移搜索主流程：Initial Search -> Drift Loop -> Synthesis
        """
        logging.info(f"启动漂移检索 (Drift Search): '{query}'")

        # 1. 初始检索 (Initial Retrieval)
        query_vector = self._embed_query(query)
        combined_context = self._build_local_context(query_vector)

        if not combined_context:
            logging.info("初始检索未找到内容，尝试直接回答或转为通用知识。")

        # 2. 漂移循环 (Drift Loop)
        for step in range(1, self.drift_max_steps + 1):
            logging.info(f"--- 漂移阶段 Step {step} ---")

            # 生成后续查询
            follow_ups = await self._generate_follow_up_queries(query, combined_context)

            if not follow_ups:
                logging.info("模型判定当前上下文已充足，停止漂移。")
                break

            logging.info(f"生成的漂移查询点: {follow_ups}")

            # 并行执行后续查询的向量检索
            new_contexts = []
            for follow_up in follow_ups:
                # 注意：这里简化处理，直接串行或并行调用 embedding 和 search
                # 实际生产中建议使用 asyncio.gather
                vec = self._embed_query(follow_up)
                ctx = self._build_local_context(vec)
                if ctx:
                    new_contexts.append(ctx)

            if not new_contexts:
                logging.info("后续查询未检索到新内容，停止漂移。")
                break

            # 合并上下文 (简单拼接，实际可做去重)
            # 注意 Token 限制，这里做简单截断保护
            combined_context += "\n\n" + "\n\n".join(new_contexts)
            if len(combined_context) > self.max_context_tokens * 4:  # 粗略字符限制
                combined_context = combined_context[:self.max_context_tokens * 4]
                logging.info("上下文过长，已截断。")
                break

        # 3. 最终合成 (Final Synthesis)
        # 复用 LocalQueryHandler 的模板逻辑，但使用累积的 combined_context
        logging.info("正在生成最终漂移搜索答案...")

        if self.search_config:
            constraints = "允许结合通用知识，但必须优先基于上下文。"
        else:
            constraints = "严格基于提供的上下文回答，禁止编造。"

        # 动态构建 Prompt
        template = Template(self.local_prompt_template)
        prompt = template.safe_substitute(
            context_data=combined_context,
            query=query,
            constraints=constraints
        )

        response = await self.generate_async_wrapper(prompt=prompt)
        return self._resolve_chunk_citations(response.text)


class GraphRouter:
    """
    智能路由器：负责意图识别和分发。
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = create_gemini_client(os.environ.get("GEMINI_API_KEY"), config["proxy"])
        self.model_name = config["query"]["generation_model"]  # 复用生成模型进行分类

        # 初始化子处理器
        self.global_handler = GlobalQueryHandler(config)
        self.drift_handler = DriftSearchHandler(config)  # 使用增强版的 Drift Handler

    async def route_and_answer(self, query: str) -> str:
        """
        路由并回答问题的入口函数。
        """
        # 1. 意图分类
        intent = await self._classify_intent(query)

        # 2. 分发执行
        if intent == "GLOBAL":
            logging.info(f"路由判定: 全局查询 (Global) -> '{query}'")
            return await self.global_handler.answer_query(query)
        else:
            logging.info(f"路由判定: 局部/漂移查询 (Local/Drift) -> '{query}'")
            # 这里调用增强的 Drift Search，而不是普通的 Local Search
            return await self.drift_handler.perform_drift_search(query)

    async def _classify_intent(self, query: str) -> Literal["GLOBAL", "LOCAL"]:
        """
        使用 LLM 判断查询意图。
        Global: 宏观、摘要、全数据集范围 (e.g. "主要的主题是什么?")
        Local: 具体实体、特定关系、细节查询 (e.g. "洋甘菊有什么功效?")
        """
        prompt = f"""
        You are a query intent classifier for a Knowledge Graph RAG system.

        Query: "{query}"

        Classify this query into one of two categories:
        1. "GLOBAL": The user asks for a summary, main themes, or an overview of the entire dataset/document collection. The answer requires aggregating information from many clusters.
        2. "LOCAL": The user asks about specific entities (people, places, concepts), their attributes, or specific relationships between them. The answer requires finding specific nodes in the graph.

        Return ONLY the word "GLOBAL" or "LOCAL". Do not add punctuation.
        """

        try:
            # 使用同步调用包装
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.models.generate_content(model=self.model_name, contents=prompt)
            )
            result = response.text.strip().upper()
            if "GLOBAL" in result: return "GLOBAL"
            return "LOCAL"  # 默认为 Local (更安全)
        except Exception as e:
            logging.error(f"意图分类失败: {e}，默认使用 LOCAL 模式。")
            return "LOCAL"


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="GraphRAG 智能路由与漂移检索")
    parser.add_argument("query", type=str, nargs='?', default="", help="输入问题")
    args = parser.parse_args()

    try:
        # 加载配置
        config = load_config()
        setup_logging(config)

        # 初始化路由器
        router = GraphRouter(config)

        if args.query:
            print(f"正在处理: {args.query}")
            answer = await router.route_and_answer(args.query)
            logging.info(f"最终答案:\n{answer}")
            print("\n--- 最终答案 ---\n")
            print(answer)
        else:
            print("进入交互模式 (输入 exit 退出)")
            while True:
                q = input("\n问题: ")
                if q.lower() in ["exit", "quit"]: break
                answer = await router.route_and_answer(q)
                logging.info(f"问题: {q}\n答案:\n{answer}")
                print(f"\n>>> 答案:\n{answer}\n")

    except Exception as e:
        logging.critical(f"程序运行错误: {e}", exc_info=True)
        print(f"错误: {e}")


if __name__ == "__main__":
    asyncio.run(main())