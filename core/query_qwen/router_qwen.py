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
import gc
from typing import Dict, Any, List, Literal
from string import Template
import yaml
from pathlib import Path
import os

from openai import OpenAI

# 复用现有模块
from global_query_qwen import GlobalQueryHandler
from local_query_qwen import LocalQueryHandler
from reasoning_query_qwen import ReasoningQueryHandler

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
    relative_log_path = log_config.get("log_file", "logs/router_qwen.log")
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
    注意：此类依赖 LocalQueryHandler 已迁移至 OpenAI SDK。
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.drift_k_followups = config["query"].get("drift_k_followups", 2)
        self.drift_max_steps = config["query"].get("drift_max_steps", 2)
        # client 已在父类 LocalQueryHandler 中初始化为 OpenAI Client

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
            # 使用父类的 generate_async_wrapper (现已改为 OpenAI 兼容逻辑)
            # 或者直接调用 OpenAI
            response_text = await self.generate_async_wrapper(prompt=prompt)

            # 简单的 JSON 提取
            match = re.search(r'\[.*\]', response_text, re.DOTALL)
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
        # _embed_query 和 _build_local_context 均继承自 LocalQueryHandler
        query_vector = self._embed_query(query)
        combined_context = self._build_local_context(query_vector)

        if not combined_context:
            logging.info("初始检索未找到内容，尝试直接回答或转为通用知识。")
            combined_context = ""  # 确保非 None

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
                # 验证后续查询不为空
                if not follow_up or not follow_up.strip():
                    logging.warning(f"跳过空的后续查询")
                    continue

                follow_up = follow_up.strip()

                # 实际生产中建议使用 asyncio.gather 优化
                try:
                    vec = self._embed_query(follow_up)
                    ctx = self._build_local_context(vec)
                    if ctx:
                        new_contexts.append(ctx)
                except Exception as e:
                    logging.error(f"处理后续查询 '{follow_up}' 时出错: {e}")

            if not new_contexts:
                logging.info("后续查询未检索到新内容，停止漂移。")
                break

            # 合并上下文（内存优化：更严格的限制）
            new_context_text = "\n\n".join(new_contexts)

            # 估算 token 数量（粗略：1 token ≈ 4 字符）
            current_tokens = len(combined_context) // 4
            new_tokens = len(new_context_text) // 4

            # 设置更严格的上限：max_context_tokens * 2（而非 * 4）
            max_total_tokens = self.max_context_tokens * 2

            if current_tokens + new_tokens > max_total_tokens:
                # 需要截断
                available_tokens = max_total_tokens - current_tokens
                if available_tokens <= 0:
                    logging.info(f"上下文已达上限 ({current_tokens} tokens)，停止漂移。")
                    break

                # 截断新上下文
                available_chars = available_tokens * 4
                new_context_text = new_context_text[:available_chars]
                logging.info(f"新上下文被截断到 {available_tokens} tokens，总计 {current_tokens + available_tokens} tokens")

            combined_context += "\n\n" + new_context_text

            # 最终安全检查
            if len(combined_context) > max_total_tokens * 4:
                combined_context = combined_context[:max_total_tokens * 4]
                logging.warning(f"上下文超过限制，强制截断到 {max_total_tokens} tokens")
                break

        # 3. 最终合成 (Final Synthesis)
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

        response_text = await self.generate_async_wrapper(prompt=prompt)

        # 内存优化：显式释放大对象并触发垃圾回收
        del combined_context
        gc.collect()

        return self._resolve_chunk_citations(response_text)


class GraphRouter:
    """
    智能路由器：负责意图识别和分发。
    支持四种查询模式：
    1. GLOBAL: 全局查询（摘要、主题等）
    2. LOCAL: 局部查询（特定实体、关系）
    3. REASONING: 推理查询（多跳推理、因果关系）
    4. DRIFT: 漂移搜索（需要上下文扩展的查询）

    内存优化：使用单例模式共享数据，避免重复加载
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # 修改：读取 QWEN_API_KEY，不设置代理
        self.api_key = os.environ.get("QWEN_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            logging.warning("未找到 QWEN_API_KEY 或 GEMINI_API_KEY 环境变量")

        # 修改：初始化 OpenAI 客户端 (兼容阿里云百炼)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        self.model_name = config["query"]["generation_model"]  # 复用生成模型进行分类

        # 内存优化：使用共享的 DriftSearchHandler，避免重复加载数据
        # DriftSearchHandler 继承自 LocalQueryHandler，已包含所有局部查询功能
        logging.info("💡 [内存优化] 使用共享数据加载器初始化处理器...")

        # 只初始化一次 DriftHandler（包含 Local 功能）
        self.drift_handler = DriftSearchHandler(config)

        # Global handler 需要单独初始化（使用不同的数据）
        self.global_handler = GlobalQueryHandler(config)

        # 初始化推理处理器（智能检测模型）
        try:
            model_path = PROJECT_ROOT / config.get('reasoning', {}).get('output', {}).get('model_path', 'data/reasoning/model.pt')
            load_model = model_path.exists()

            if load_model:
                logging.info(f"检测到推理模型: {model_path}")
                # 注意：ReasoningQueryHandler 需要 GraphData 对象（dataclass），
                # 而 drift_handler.graph_data 是 dict 类型（JSON格式），不能共享
                # 因此让 ReasoningQueryHandler 自行加载其所需的 GraphData
                logging.info("正在加载推理处理器...")
                self.reasoning_handler = ReasoningQueryHandler(
                    config,
                    load_trained_model=True,
                    shared_graph_data=None  # 不共享，让其自行加载
                )
                self.reasoning_enabled = True
            else:
                logging.warning(f"未找到推理模型: {model_path}，推理功能将被禁用")
                self.reasoning_handler = None
                self.reasoning_enabled = False
        except Exception as e:
            logging.error(f"加载推理处理器失败: {e}，推理功能将被禁用")
            self.reasoning_handler = None
            self.reasoning_enabled = False

        logging.info(f"✅ 路由器初始化完成")
        logging.info(f"   - 启用处理器: Global, Local, Drift" +
                    (", Reasoning" if self.reasoning_enabled else ""))
        if self.reasoning_enabled:
            logging.info(f"   - 推理处理器: 已启用")

    async def route_and_answer(self, query: str) -> str:
        """
        路由并回答问题的入口函数。
        使用 CoT (Chain of Thought) 技术进行智能分类和方法选择。
        """
        # 输入验证
        if not query or not query.strip():
            logging.error("❌ 收到空查询字符串")
            return "错误：查询不能为空，请提供有效的问题。"

        query = query.strip()  # 清理空白字符
        logging.info(f"📥 收到查询: '{query[:100]}...'")

        # 1. CoT 意图分类（包含推理模式）
        classification_result = await self._cot_classify_intent(query)
        intent = classification_result['intent']
        reasoning = classification_result['reasoning']
        method = classification_result.get('method', 'ppr')

        logging.info(f"CoT 分类结果:")
        logging.info(f"  意图: {intent}")
        logging.info(f"  推理: {reasoning}")
        if intent == "REASONING":
            logging.info(f"  推理方法: {method}")

        # 2. 分发执行
        if intent == "GLOBAL":
            logging.info(f"路由判定: 全局查询 (Global) -> '{query}'")
            return await self.global_handler.answer_query(query)

        elif intent == "REASONING":
            if not self.reasoning_enabled:
                logging.warning("推理功能未启用，降级为漂移搜索模式")
                return await self.drift_handler.perform_drift_search(query)

            logging.info(f"路由判定: 推理查询 (Reasoning) with method={method} -> '{query}'")
            # 使用同步方法，但包装在 executor 中
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.reasoning_handler.query(query, method=method, include_llm_answer=True)
            )
            # 提取答案文本
            return result.get('answer', '未能生成答案')

        elif intent == "DRIFT":
            logging.info(f"路由判定: 漂移搜索 (Drift) -> '{query}'")
            return await self.drift_handler.perform_drift_search(query)

        else:  # LOCAL
            logging.info(f"路由判定: 局部查询 (Local) -> '{query}'")
            # 使用 DriftHandler 的基础 LocalQueryHandler 功能
            # 直接调用一次检索即可
            query_vector = self.drift_handler._embed_query(query)
            context = self.drift_handler._build_local_context(query_vector)

            # 使用模板生成答案
            from string import Template
            template = Template(self.drift_handler.local_prompt_template)
            prompt = template.safe_substitute(
                context_data=context,
                query=query,
                constraints="严格基于提供的上下文回答，禁止编造。"
            )

            response_text = await self.drift_handler.generate_async_wrapper(prompt=prompt)
            return self.drift_handler._resolve_chunk_citations(response_text)

    async def _cot_classify_intent(self, query: str) -> Dict[str, Any]:
        """
        使用 Chain of Thought (CoT) 技术进行意图分类和方法选择。

        Returns:
            Dict with keys: 'intent', 'reasoning', 'method' (for REASONING intent)
        """
        cot_prompt = f"""
You are an intelligent query classifier for a Knowledge Graph RAG system. Use Chain of Thought reasoning to classify the query.

Query: "{query}"

Think step by step:

1. **Query Analysis**: What is the user asking for?
   - Is it asking for a summary/overview of the entire dataset? (GLOBAL)
   - Is it asking about specific entities and their direct attributes? (LOCAL)
   - Does it require multi-hop reasoning or finding relationships between entities? (REASONING)
   - Does it need context expansion through multiple retrieval steps? (DRIFT)

2. **Complexity Assessment**: 
   - Simple fact lookup → LOCAL
   - Relationship discovery/causal reasoning → REASONING
   - Broad overview/summary → GLOBAL
   - Needs iterative refinement → DRIFT

3. **Keywords Identification**:
   - GLOBAL indicators: "summarize", "overview", "main themes", "in general"
   - LOCAL indicators: "what is", "define", "describe [specific entity]"
   - REASONING indicators: "relationship between", "why", "how does X affect Y", "connection", "path from X to Y"
   - DRIFT indicators: "comprehensive analysis", "explore", "deep dive"

4. **Method Selection** (if REASONING):
   - PPR (Personalized PageRank): Better for discovering related entities through graph structure
   - GNN (Graph Neural Network): Better for complex multi-hop reasoning with learned patterns
   - Use PPR for: "find related", "what connects", "entities similar to"
   - Use GNN for: "why", "causal relationship", "complex interaction"

Now classify the query:

**Reasoning Process**:
[Your step-by-step analysis here]

**Classification**:
Intent: [GLOBAL/LOCAL/REASONING/DRIFT]
{f"Method: [ppr/gnn] (only if REASONING)" if "relationship" in query.lower() or "why" in query.lower() else ""}

Provide your response in this exact format:
REASONING: <your detailed reasoning>
INTENT: <GLOBAL/LOCAL/REASONING/DRIFT>
METHOD: <ppr/gnn> (only if REASONING)
"""

        try:
            loop = asyncio.get_running_loop()

            def call_classifier():
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": cot_prompt}],
                    temperature=0.3  # 适中温度，允许一定创造性
                )
                return response.choices[0].message.content

            result_text = await loop.run_in_executor(None, call_classifier)

            # 解析 CoT 输出
            reasoning = ""
            intent = "LOCAL"  # 默认
            method = "ppr"  # 默认

            lines = result_text.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith("REASONING:"):
                    reasoning = line.replace("REASONING:", "").strip()
                elif line.startswith("INTENT:"):
                    intent_str = line.replace("INTENT:", "").strip().upper()
                    if intent_str in ["GLOBAL", "LOCAL", "REASONING", "DRIFT"]:
                        intent = intent_str
                elif line.startswith("METHOD:"):
                    method_str = line.replace("METHOD:", "").strip().lower()
                    if method_str in ["ppr", "gnn"]:
                        method = method_str

            # 如果没有推理模型，自动降级 REASONING 为 DRIFT
            if intent == "REASONING" and not self.reasoning_enabled:
                logging.info("检测到 REASONING 意图，但推理模型未启用，降级为 DRIFT")
                intent = "DRIFT"

            return {
                'intent': intent,
                'reasoning': reasoning,
                'method': method
            }

        except Exception as e:
            logging.error(f"CoT 意图分类失败: {e}，使用简单分类降级")
            # 降级为简单分类
            return await self._simple_classify_intent(query)

    async def _simple_classify_intent(self, query: str) -> Dict[str, Any]:
        """
        简单的意图分类（作为 CoT 的降级方案）。
        """
        prompt = f"""
Classify this query into one category:
1. "GLOBAL": Summary/overview of entire dataset
2. "LOCAL": Specific entity attributes
3. "REASONING": Multi-hop reasoning or relationships
4. "DRIFT": Needs context expansion

Query: "{query}"

Return ONLY one word: GLOBAL, LOCAL, REASONING, or DRIFT
"""

        try:
            loop = asyncio.get_running_loop()

            def call_classifier():
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                return response.choices[0].message.content

            result_text = await loop.run_in_executor(None, call_classifier)
            result = result_text.strip().upper()

            # 检测关键词以选择方法
            method = "ppr"  # 默认
            if "why" in query.lower() or "cause" in query.lower():
                method = "gnn"

            for intent in ["GLOBAL", "LOCAL", "REASONING", "DRIFT"]:
                if intent in result:
                    # 如果没有推理模型，降级
                    if intent == "REASONING" and not self.reasoning_enabled:
                        intent = "DRIFT"

                    return {
                        'intent': intent,
                        'reasoning': f"Simple classification based on keywords",
                        'method': method
                    }

            return {
                'intent': "LOCAL",
                'reasoning': "Default fallback",
                'method': "ppr"
            }

        except Exception as e:
            logging.error(f"简单分类也失败: {e}，默认使用 LOCAL 模式")
            return {
                'intent': "LOCAL",
                'reasoning': "Error fallback",
                'method': "ppr"
            }

    async def _classify_intent(self, query: str) -> Literal["GLOBAL", "LOCAL"]:
        """
        使用 LLM 判断查询意图。
        （保留用于向后兼容，现在推荐使用 _cot_classify_intent）
        """
        result = await self._simple_classify_intent(query)
        return result['intent']


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="智能路由与漂移检索")
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