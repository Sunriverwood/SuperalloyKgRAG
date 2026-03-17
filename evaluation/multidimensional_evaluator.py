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

"""
多维度评测模块 - 对比评估 Baseline 模型和 RAG 方法的效果

功能：
1. 调用 6 个 Baseline 模型和 5 种 RAG 方法获取回答
2. 使用 LLM 作为评判官，基于六个维度（正确性、全面性、多样性、直接性、赋能性、忠实性）进行排序评分
3. 支持断点续传
4. 生成详细的评测报告和对比矩阵（所有输出文件带时间戳）
"""

import argparse
import asyncio
import csv
import gc
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Dict, Any, List, Optional, Tuple

import yaml
from openai import OpenAI

# --- 项目根目录定义 ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# 导入现有模块
from evaluation.auto_evaluator import EvaluationDataLoader


def load_config(settings_filename: str = "settings.yaml") -> Dict[str, Any]:
    """加载YAML配置文件"""
    config_path = PROJECT_ROOT / "config" / settings_filename
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件 {config_path} 未找到！")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_logging(config: Dict[str, Any], log_name: str = "multidimensional_evaluation"):
    """设置日志记录器"""
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
    log_file = PROJECT_ROOT / "logs" / f"{log_name}.log"
    log_file.parent.mkdir(exist_ok=True, parents=True)

    # 移除现有处理器
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info("=" * 80)
    logging.info(f"多维度评测日志记录器设置完成: {log_file}")
    logging.info("=" * 80)


def resolve_api_key(api_key_str: str) -> str:
    """解析 API Key，支持 ${VAR} 格式的环境变量引用"""
    if not api_key_str:
        return ""
    api_key_str = api_key_str.strip()
    if api_key_str.startswith("${") and api_key_str.endswith("}"):
        var_name = api_key_str[2:-1].strip()
        env_val = os.getenv(var_name)
        if not env_val:
            logging.warning(f"环境变量 {var_name} 未设置")
            return api_key_str
        return env_val
    return api_key_str


# 六个评价维度（新增 correctness 作为最重要维度）
DIMENSIONS = ["correctness", "comprehensiveness", "diversity", "directness", "empowerment", "faithfulness"]
DIMENSION_NAMES_CN = {
    "correctness": "正确性",
    "comprehensiveness": "全面性",
    "diversity": "多样性",
    "directness": "直接性",
    "empowerment": "赋能性",
    "faithfulness": "忠实性"
}


class SharedHandlers:
    """
    共享的 Query Handler 管理器
    实现延迟初始化，避免重复加载数据
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._basic_rag_handler = None
        self._global_handler = None
        self._local_handler = None
        self._reasoning_handler = None
        self._router_handler = None
        self._initialized = set()

    def get_basic_rag_handler(self):
        """获取 Basic RAG Handler（延迟初始化）"""
        if "basic_rag" not in self._initialized:
            logging.info("初始化 Basic RAG Handler...")
            # 动态导入以避免 IDE 静态分析问题
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "basic_rag_qwen",
                PROJECT_ROOT / "core" / "query_qwen" / "basic_rag_qwen.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._basic_rag_handler = module.RAGQueryHandler(self.config)
            self._initialized.add("basic_rag")
        return self._basic_rag_handler

    def get_global_handler(self):
        """获取 Global Query Handler（延迟初始化）"""
        if "global" not in self._initialized:
            logging.info("初始化 Global Query Handler...")
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "global_query_qwen",
                PROJECT_ROOT / "core" / "query_qwen" / "global_query_qwen.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._global_handler = module.GlobalQueryHandler(self.config)
            self._initialized.add("global")
        return self._global_handler

    def get_local_handler(self):
        """获取 Local Query Handler（延迟初始化）"""
        if "local" not in self._initialized:
            logging.info("初始化 Local Query Handler...")
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "local_query_qwen",
                PROJECT_ROOT / "core" / "query_qwen" / "local_query_qwen.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._local_handler = module.LocalQueryHandler(self.config)
            self._initialized.add("local")
        return self._local_handler

    def get_reasoning_handler(self):
        """获取 Reasoning Query Handler（延迟初始化）"""
        if "reasoning" not in self._initialized:
            logging.info("初始化 Reasoning Query Handler...")
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "reasoning_query_qwen",
                    PROJECT_ROOT / "core" / "query_qwen" / "reasoning_query_qwen.py"
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                model_path = PROJECT_ROOT / self.config.get('reasoning', {}).get('output', {}).get(
                    'model_path', 'data/reasoning/model.pt')
                if model_path.exists():
                    self._reasoning_handler = module.ReasoningQueryHandler(
                        self.config,
                        load_trained_model=True,
                        shared_graph_data=None
                    )
                else:
                    logging.warning(f"推理模型文件不存在: {model_path}，Reasoning 方法将不可用")
                    self._reasoning_handler = None
            except Exception as e:
                logging.error(f"初始化 Reasoning Handler 失败: {e}")
                self._reasoning_handler = None
            self._initialized.add("reasoning")
        return self._reasoning_handler

    def get_router_handler(self):
        """获取 Router Handler（延迟初始化）"""
        if "router" not in self._initialized:
            logging.info("初始化 Router Handler...")
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "router_qwen",
                PROJECT_ROOT / "core" / "query_qwen" / "router_qwen.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._router_handler = module.GraphRouter(self.config)
            self._initialized.add("router")
        return self._router_handler

    def cleanup(self):
        """释放资源"""
        self._basic_rag_handler = None
        self._global_handler = None
        self._local_handler = None
        self._reasoning_handler = None
        self._router_handler = None
        self._initialized.clear()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class MultidimensionalEvaluator:
    """多维度评测器"""

    def __init__(
            self,
            config: Dict[str, Any],
            enabled_methods: Optional[List[str]] = None,
            max_concurrency: int = 5
    ):
        """
        初始化多维度评测器

        Args:
            config: 配置字典
            enabled_methods: 启用的方法列表，None 表示全部启用
            max_concurrency: 最大并发数
        """
        self.config = config
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)

        # 运行时间戳（用于输出文件命名）
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 数据加载器
        self.data_loader = EvaluationDataLoader()

        # 从配置文件读取多维度评测配置
        multi_eval_config = config.get("multidimensional_evaluation", {})

        # 输出目录（从配置读取，支持相对路径和绝对路径）
        answers_dir_path = multi_eval_config.get("answers_output_dir", "data/answers/multidimensional_evaluation")
        self.answers_dir = PROJECT_ROOT / answers_dir_path if not Path(answers_dir_path).is_absolute() else Path(answers_dir_path)
        self.answers_dir.mkdir(exist_ok=True, parents=True)

        reports_dir_path = multi_eval_config.get("reports_output_dir", "data/reports/multidimensional_evaluation")
        self.reports_dir = PROJECT_ROOT / reports_dir_path if not Path(reports_dir_path).is_absolute() else Path(reports_dir_path)
        self.reports_dir.mkdir(exist_ok=True, parents=True)

        # 断点文件
        self.checkpoint_file = self.reports_dir / "checkpoint.json"
        self.enable_checkpoint = multi_eval_config.get("enable_checkpoint", True)

        # 加载评判 Prompt（从配置读取路径）
        judge_prompt_rel_path = multi_eval_config.get("judge_prompt_path", "config/prompts/multidimensional_judge.md")
        judge_prompt_path = PROJECT_ROOT / judge_prompt_rel_path if not Path(judge_prompt_rel_path).is_absolute() else Path(judge_prompt_rel_path)
        if judge_prompt_path.exists():
            with open(judge_prompt_path, 'r', encoding='utf-8') as f:
                self.judge_prompt_template = f.read()
        else:
            raise FileNotFoundError(f"评判 Prompt 文件不存在: {judge_prompt_path}")

        # 初始化评判 LLM 客户端
        eval_llm_config = config.get("evaluation", {}).get("llm", {})
        self.judge_client = OpenAI(
            api_key=resolve_api_key(eval_llm_config.get("api_key", "${QWEN_API_KEY}")),
            base_url=eval_llm_config.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        )
        self.judge_model = eval_llm_config.get("model", "qwen3-max")

        # Baseline 模型配置
        self.baseline_models = config.get("baseline", {}).get("models", [])

        # RAG 方法列表
        self.rag_methods = ["basic_rag", "global", "local", "reasoning", "router"]

        # 确定启用的方法（优先使用参数，其次配置文件，最后全部启用）
        all_methods = [m["name"] for m in self.baseline_models] + self.rag_methods
        config_enabled_methods = multi_eval_config.get("enabled_methods", [])

        if enabled_methods:
            # 命令行参数优先
            self.enabled_methods = [m for m in enabled_methods if m in all_methods]
        elif config_enabled_methods:
            # 其次使用配置文件中的设置
            self.enabled_methods = [m for m in config_enabled_methods if m in all_methods]
        else:
            # 默认全部启用
            self.enabled_methods = all_methods

        logging.info(f"启用的评测方法: {self.enabled_methods}")

        # 共享 Handler 管理器（延迟初始化）
        self.shared_handlers = SharedHandlers(config)

        # 重试配置
        self.retry_count = multi_eval_config.get("retry_count", config.get("evaluation", {}).get("retry_count", 2))

        # API 请求间隔（秒），用于避免限流
        self.request_interval = multi_eval_config.get("request_interval", 1.0)

        # 性能统计
        self.stats = {
            "total_questions": 0,
            "completed_questions": 0,
            "failed_questions": 0,
            "total_answers": 0,
            "failed_answers": 0,
            "judge_failures": 0
        }

    def _needs_proxy(self, model_name: str) -> bool:
        """判断模型是否需要代理"""
        model_lower = model_name.lower()
        return "gemini" in model_lower or "gpt" in model_lower or "chatgpt" in model_lower

    async def _get_baseline_answer(self, question: str, model_config: Dict[str, Any]) -> Tuple[str, float]:
        """
        调用 Baseline 模型获取回答

        Returns:
            (answer, latency_seconds)
        """
        async with self.semaphore:
            start_time = time.time()
            try:
                api_key = resolve_api_key(model_config.get("api_key", ""))
                base_url = model_config.get("base_url")
                model_name = model_config.get("name")

                # 代理处理
                if self._needs_proxy(model_name):
                    proxy = self.config.get("proxy")
                    if proxy:
                        os.environ["HTTP_PROXY"] = proxy
                        os.environ["HTTPS_PROXY"] = proxy
                else:
                    os.environ.pop("HTTP_PROXY", None)
                    os.environ.pop("HTTPS_PROXY", None)

                client = OpenAI(api_key=api_key, base_url=base_url)

                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=model_name,
                    messages=[{"role": "user", "content": question}],
                    temperature=model_config.get("temperature", 0.1),
                    max_tokens=model_config.get("max_tokens", 2000)
                )

                answer = response.choices[0].message.content.strip()
                latency = time.time() - start_time
                return answer, latency

            except Exception as e:
                logging.error(f"Baseline 模型 {model_config.get('name')} 获取回答失败: {e}")
                return f"[ERROR] {str(e)}", time.time() - start_time

    async def _get_rag_answer(self, question: str, method: str) -> Tuple[str, float]:
        """
        调用 RAG 方法获取回答

        Returns:
            (answer, latency_seconds)
        """
        async with self.semaphore:
            start_time = time.time()
            try:
                if method == "basic_rag":
                    handler = self.shared_handlers.get_basic_rag_handler()
                    answer = handler.answer_query(question)

                elif method == "global":
                    handler = self.shared_handlers.get_global_handler()
                    answer = await handler.answer_query(question)

                elif method == "local":
                    handler = self.shared_handlers.get_local_handler()
                    answer = await handler.answer_query(question)

                elif method == "reasoning":
                    handler = self.shared_handlers.get_reasoning_handler()
                    if handler is None:
                        return "[ERROR] Reasoning handler 不可用", time.time() - start_time
                    loop = asyncio.get_running_loop()

                    def _run_reasoning():
                        return handler.query(question, include_llm_answer=True)

                    result = await loop.run_in_executor(None, _run_reasoning)
                    answer = result.get('answer', '未能生成答案')

                elif method == "router":
                    handler = self.shared_handlers.get_router_handler()
                    answer = await handler.route_and_answer(question)

                else:
                    return f"[ERROR] 未知的 RAG 方法: {method}", time.time() - start_time

                latency = time.time() - start_time
                return answer, latency

            except Exception as e:
                logging.error(f"RAG 方法 {method} 获取回答失败: {e}")
                import traceback
                traceback.print_exc()
                return f"[ERROR] {str(e)}", time.time() - start_time

    async def _get_all_answers(self, question: str) -> Dict[str, Dict[str, Any]]:
        """
        获取所有启用方法的回答
        改为顺序执行并添加请求间隔，避免 API 限流

        Returns:
            {method_name: {"answer": str, "latency": float, "error": bool}}
        """
        results = {}

        for idx, method in enumerate(self.enabled_methods):
            # 添加请求间隔（第一个请求不需要等待）
            if idx > 0 and self.request_interval > 0:
                await asyncio.sleep(self.request_interval)

            try:
                # 检查是否为 Baseline 模型
                baseline_config = next((m for m in self.baseline_models if m["name"] == method), None)
                if baseline_config:
                    answer, latency = await self._get_baseline_answer(question, baseline_config)
                elif method in self.rag_methods:
                    answer, latency = await self._get_rag_answer(question, method)
                else:
                    answer, latency = f"[ERROR] 未知方法: {method}", 0

                is_error = answer.startswith("[ERROR]")
                results[method] = {
                    "answer": answer,
                    "latency": latency,
                    "error": is_error
                }

                if is_error:
                    self.stats["failed_answers"] += 1
                self.stats["total_answers"] += 1

                logging.info(f"  方法 {method} 完成，耗时 {latency:.2f}s")

            except Exception as e:
                results[method] = {
                    "answer": f"[ERROR] {str(e)}",
                    "latency": 0,
                    "error": True
                }
                self.stats["failed_answers"] += 1
                logging.error(f"  方法 {method} 失败: {e}")

        return results

    def _parse_judge_response(self, response_text: str, method_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        解析评判 LLM 的响应

        Returns:
            {dimension: {"ranking": [method1, method2, ...], "scores": {method: score}}}
        """
        result = {}

        # 尝试提取 JSON
        json_match = re.search(r'\{[\s\S]*}', response_text)
        if json_match:
            try:
                parsed = json.loads(json_match.group())

                for dim in DIMENSIONS:
                    if dim in parsed:
                        ranking = parsed[dim].get("ranking", [])
                        ties = parsed[dim].get("ties", [])

                        # 验证方法名
                        valid_ranking = [m for m in ranking if m in method_names]

                        # 计算分数：排名第1得N分，排名第N得1分
                        n = len(method_names)
                        scores = {}

                        # 处理并列情况
                        tie_groups: Dict[str, Optional[Tuple]] = {m: None for m in method_names}
                        for tie_group in ties:
                            for m in tie_group:
                                if m in tie_groups:
                                    tie_groups[m] = tuple(tie_group)

                        # 分配分数
                        rank_position = 0
                        processed = set()
                        for method in valid_ranking:
                            if method in processed:
                                continue

                            tie_group = tie_groups.get(method)
                            if tie_group is not None:
                                # 并列的方法得相同分数（取平均）
                                group_size = len([m for m in tie_group if m in valid_ranking])
                                avg_score = sum(n - rank_position - i for i in range(group_size)) / group_size
                                for m in tie_group:
                                    if m in method_names:
                                        scores[m] = avg_score
                                        processed.add(m)
                                rank_position += group_size
                            else:
                                scores[method] = n - rank_position
                                processed.add(method)
                                rank_position += 1

                        # 未出现在排名中的方法得最低分
                        for m in method_names:
                            if m not in scores:
                                scores[m] = 1

                        result[dim] = {
                            "ranking": valid_ranking,
                            "scores": scores,
                            "ties": ties
                        }

                return result

            except json.JSONDecodeError as e:
                logging.warning(f"JSON 解析失败: {e}")

        # JSON 解析失败，尝试正则解析
        logging.warning("尝试使用正则表达式解析评判结果...")
        for dim in DIMENSIONS:
            pattern = rf'{dim}[^:]*:\s*["\']?ranking["\']?\s*:\s*\[(.*?)\]'
            match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
            if match:
                ranking_str = match.group(1)
                # 提取方法名
                methods_found = re.findall(r'["\']([^"\']+)["\']', ranking_str)
                valid_ranking = [m for m in methods_found if m in method_names]

                n = len(method_names)
                scores = {m: n - i for i, m in enumerate(valid_ranking)}
                for m in method_names:
                    if m not in scores:
                        scores[m] = 1

                result[dim] = {
                    "ranking": valid_ranking,
                    "scores": scores,
                    "ties": []
                }

        return result

    async def _judge_answers(
            self,
            question: str,
            ground_truth: str,
            answers: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        使用 LLM 评判所有回答

        Returns:
            {
                "dimensions": {dim: {"ranking": [...], "scores": {...}}},
                "raw_response": str,
                "parse_success": bool
            }
        """
        # 过滤掉错误的回答
        valid_answers = {k: v for k, v in answers.items() if not v.get("error", False)}
        if len(valid_answers) < 2:
            logging.warning("有效回答少于2个，跳过评判")
            return {
                "dimensions": {},
                "raw_response": "",
                "parse_success": False,
                "skip_reason": "有效回答不足"
            }

        # 构建回答文本
        answers_text = ""
        method_names = list(valid_answers.keys())
        for i, (method, data) in enumerate(valid_answers.items(), 1):
            answers_text += f"\n**回答 {i} (方法: {method})**:\n{data['answer']}\n"

        # 填充 Prompt
        prompt = Template(self.judge_prompt_template).safe_substitute(
            question=question,
            ground_truth=ground_truth,
            answers=answers_text
        )

        # 评判前等待一下，避免限流
        if self.request_interval > 0:
            await asyncio.sleep(self.request_interval)

        # 调用评判 LLM（带重试）
        raw_response = ""
        for attempt in range(self.retry_count + 1):
            try:
                response = await asyncio.to_thread(
                    self.judge_client.chat.completions.create,
                    model=self.judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=2000
                )
                raw_response = response.choices[0].message.content
                break
            except Exception as e:
                logging.warning(f"评判 LLM 调用失败 (尝试 {attempt + 1}/{self.retry_count + 1}): {e}")
                if attempt == self.retry_count:
                    self.stats["judge_failures"] += 1
                    return {
                        "dimensions": {},
                        "raw_response": str(e),
                        "parse_success": False,
                        "error": str(e)
                    }
                await asyncio.sleep(2 ** attempt)  # 指数退避

        # 解析响应
        parsed = self._parse_judge_response(raw_response, method_names)
        parse_success = len(parsed) == len(DIMENSIONS)

        if not parse_success:
            logging.warning(f"评判结果解析不完整，仅解析到 {len(parsed)}/{len(DIMENSIONS)} 个维度")
            self.stats["judge_failures"] += 1

        return {
            "dimensions": parsed,
            "raw_response": raw_response,
            "parse_success": parse_success
        }

    def _load_checkpoint(self) -> Dict[str, Any]:
        """加载断点"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"completed_ids": [], "last_updated": None}

    def _save_checkpoint(self, completed_ids: List[int]):
        """保存断点"""
        checkpoint = {
            "completed_ids": completed_ids,
            "last_updated": datetime.now().isoformat(),
            "stats": self.stats
        }
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)

    def _save_answer(self, method: str, question_id: int, question: str, answer: str, latency: float):
        """保存单个回答到文件"""
        answer_file = self.answers_dir / f"{method}_answers_{self.run_timestamp}.jsonl"
        record = {
            "question_id": question_id,
            "question": question,
            "answer": answer,
            "latency_seconds": latency,
            "timestamp": datetime.now().isoformat()
        }
        with open(answer_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    def _save_detailed_result(self, result: Dict[str, Any]):
        """保存详细结果"""
        result_file = self.reports_dir / f"detailed_results_{self.run_timestamp}.jsonl"
        with open(result_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    async def evaluate_single(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """评测单道题目"""
        question_id = item.get("id")
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")
        difficulty = item.get("difficulty", "L1")
        question_type = item.get("type", "")
        domain = item.get("domain", "")
        theme = item.get("theme", "")

        logging.info(f"[Q{question_id}] 开始评测: {question[:50]}...")

        # 获取所有回答
        start_time = time.time()
        answers = await self._get_all_answers(question)
        answer_time = time.time() - start_time

        logging.info(f"[Q{question_id}] 获取 {len(answers)} 个回答耗时: {answer_time:.2f}s")

        # 保存回答
        for method, data in answers.items():
            self._save_answer(method, question_id, question, data["answer"], data["latency"])

        # 评判回答
        judge_start = time.time()
        judge_result = await self._judge_answers(question, ground_truth, answers)
        judge_time = time.time() - judge_start

        logging.info(f"[Q{question_id}] 评判耗时: {judge_time:.2f}s, 解析成功: {judge_result['parse_success']}")

        # 构建结果
        result = {
            "question_id": question_id,
            "question": question,
            "ground_truth": ground_truth,
            "difficulty": difficulty,
            "type": question_type,
            "domain": domain,
            "theme": theme,
            "answers": {k: {"answer": v["answer"], "latency": v["latency"], "error": v["error"]}
                       for k, v in answers.items()},
            "judge_result": judge_result,
            "answer_time_seconds": answer_time,
            "judge_time_seconds": judge_time,
            "timestamp": datetime.now().isoformat()
        }

        # 保存详细结果
        self._save_detailed_result(result)

        return result

    def generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成评测汇总报告"""
        summary = {
            "report_type": "multidimensional_evaluation",
            "generated_at": datetime.now().isoformat(),
            "total_questions": len(results),
            "enabled_methods": self.enabled_methods,
            "dimension_scores": {},
            "overall_ranking": {},
            "by_difficulty": {},
            "stats": self.stats
        }

        # 按维度统计分数
        dimension_totals = {dim: {method: [] for method in self.enabled_methods} for dim in DIMENSIONS}

        for result in results:
            judge_result = result.get("judge_result", {})
            dimensions = judge_result.get("dimensions", {})

            for dim, data in dimensions.items():
                scores = data.get("scores", {})
                for method, score in scores.items():
                    if method in dimension_totals[dim]:
                        dimension_totals[dim][method].append(score)

        # 计算平均分
        for dim in DIMENSIONS:
            summary["dimension_scores"][dim] = {}
            for method in self.enabled_methods:
                scores = dimension_totals[dim][method]
                if scores:
                    avg_score = sum(scores) / len(scores)
                    summary["dimension_scores"][dim][method] = {
                        "average_score": round(avg_score, 3),
                        "count": len(scores)
                    }

        # 计算综合排名（所有维度平均）
        overall_scores = {method: [] for method in self.enabled_methods}
        for dim in DIMENSIONS:
            for method in self.enabled_methods:
                if method in summary["dimension_scores"][dim]:
                    overall_scores[method].append(
                        summary["dimension_scores"][dim][method]["average_score"]
                    )

        overall_averages = {}
        for method, scores in overall_scores.items():
            if scores:
                overall_averages[method] = round(sum(scores) / len(scores), 3)

        # 排序
        sorted_methods = sorted(overall_averages.items(), key=lambda x: x[1], reverse=True)
        summary["overall_ranking"] = {
            method: {"rank": i + 1, "average_score": score}
            for i, (method, score) in enumerate(sorted_methods)
        }

        # 按难度统计
        difficulty_groups = {}
        for result in results:
            diff = result.get("difficulty", "unknown")
            if diff not in difficulty_groups:
                difficulty_groups[diff] = []
            difficulty_groups[diff].append(result)

        for diff, diff_results in difficulty_groups.items():
            summary["by_difficulty"][diff] = {
                "count": len(diff_results),
                "dimension_scores": {}
            }
            for dim in DIMENSIONS:
                dim_scores = {method: [] for method in self.enabled_methods}
                for result in diff_results:
                    dimensions = result.get("judge_result", {}).get("dimensions", {})
                    if dim in dimensions:
                        for method, score in dimensions[dim].get("scores", {}).items():
                            if method in dim_scores:
                                dim_scores[method].append(score)

                summary["by_difficulty"][diff]["dimension_scores"][dim] = {
                    method: round(sum(scores) / len(scores), 3) if scores else None
                    for method, scores in dim_scores.items()
                }

        return summary

    def generate_comparison_matrix(self, results: List[Dict[str, Any]]) -> str:
        """生成对比矩阵 CSV"""
        csv_file = self.reports_dir / f"comparison_matrix_{self.run_timestamp}.csv"

        # 构建数据
        headers = ["Method"] + [DIMENSION_NAMES_CN[d] for d in DIMENSIONS] + ["Overall"]

        rows = []
        for method in self.enabled_methods:
            row = [method]
            total_scores = []

            for dim in DIMENSIONS:
                scores = []
                for result in results:
                    dimensions = result.get("judge_result", {}).get("dimensions", {})
                    if dim in dimensions and method in dimensions[dim].get("scores", {}):
                        scores.append(dimensions[dim]["scores"][method])

                if scores:
                    avg = sum(scores) / len(scores)
                    row.append(f"{avg:.3f}")
                    total_scores.append(avg)
                else:
                    row.append("N/A")

            if total_scores:
                overall = sum(total_scores) / len(total_scores)
                row.append(f"{overall:.3f}")
            else:
                row.append("N/A")

            rows.append(row)

        # 写入 CSV
        with open(csv_file, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)

        logging.info(f"对比矩阵已保存: {csv_file}")
        return str(csv_file)

    async def run(
            self,
            difficulty: Optional[str] = None,
            question_ids: Optional[List[int]] = None,
            resume: bool = True,
            filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        运行完整评测

        Args:
            difficulty: 指定难度级别 (L12/L3/L4)，None 表示全部
            question_ids: 指定题目 ID 列表
            resume: 是否从断点继续
            filename: 指定评测文件名 (如 'hard.json')
        """
        # 加载题目
        questions = self.data_loader.load_questions(difficulty=difficulty, question_ids=question_ids, filename=filename)
        self.stats["total_questions"] = len(questions)

        logging.info(f"共加载 {len(questions)} 道题目")

        # 加载断点（同时考虑 resume 参数和配置文件中的 enable_checkpoint）
        use_checkpoint = resume and self.enable_checkpoint
        checkpoint = self._load_checkpoint() if use_checkpoint else {"completed_ids": []}
        completed_ids = set(checkpoint.get("completed_ids", []))

        # 过滤已完成的题目
        pending_questions = [q for q in questions if q["id"] not in completed_ids]
        logging.info(f"待评测题目: {len(pending_questions)} 道 (已完成: {len(completed_ids)} 道)")

        results = []
        all_completed_ids = list(completed_ids)

        # 逐题评测
        for i, item in enumerate(pending_questions, 1):
            try:
                result = await self.evaluate_single(item)
                results.append(result)
                all_completed_ids.append(item["id"])
                self.stats["completed_questions"] += 1

                # 保存断点（如果启用了断点功能）
                if self.enable_checkpoint:
                    self._save_checkpoint(all_completed_ids)

                logging.info(f"进度: {i}/{len(pending_questions)} ({100 * i / len(pending_questions):.1f}%)")

                # 定期释放内存
                if i % 50 == 0:
                    gc.collect()
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except ImportError:
                        pass

            except Exception as e:
                logging.error(f"题目 {item['id']} 评测失败: {e}")
                self.stats["failed_questions"] += 1
                import traceback
                traceback.print_exc()

        # 加载已有的详细结果（用于生成完整报告）
        all_results = []
        detailed_file = self.reports_dir / f"detailed_results_{self.run_timestamp}.jsonl"
        if detailed_file.exists():
            with open(detailed_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        all_results.append(json.loads(line))

        # 生成汇总报告
        summary = self.generate_summary(all_results)
        summary_file = self.reports_dir / f"evaluation_summary_{self.run_timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logging.info(f"汇总报告已保存: {summary_file}")

        # 生成对比矩阵
        self.generate_comparison_matrix(all_results)

        # 清理资源
        self.shared_handlers.cleanup()

        logging.info("=" * 80)
        logging.info("评测完成！")
        logging.info(f"  总题目数: {self.stats['total_questions']}")
        logging.info(f"  完成数: {self.stats['completed_questions']}")
        logging.info(f"  失败数: {self.stats['failed_questions']}")
        logging.info(f"  总回答数: {self.stats['total_answers']}")
        logging.info(f"  回答失败数: {self.stats['failed_answers']}")
        logging.info(f"  评判失败数: {self.stats['judge_failures']}")
        logging.info("=" * 80)

        return summary


async def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="多维度评测器 - 对比 Baseline 和 RAG 方法")
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="指定评测的方法（逗号分隔），例如: qwen3-max,basic_rag,global"
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default=None,
        choices=["L1", "L2", "L3", "L4"],
        help="指定难度级别（L1/L2/L3/L4），不指定则加载全部"
    )
    parser.add_argument(
        "--question-ids",
        type=str,
        default=None,
        help="指定题目 ID（逗号分隔），例如: 1,2,3"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="从断点继续执行（默认启用）"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="不从断点继续，重新开始"
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="最大并发数（默认从配置文件读取）"
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="指定评测文件名 (如 hard.json)"
    )

    args = parser.parse_args()

    # 加载配置
    config = load_config()
    setup_logging(config)

    # 解析方法列表
    enabled_methods = None
    if args.methods:
        enabled_methods = [m.strip() for m in args.methods.split(",")]

    # 解析题目 ID
    question_ids = None
    if args.question_ids:
        question_ids = [int(x.strip()) for x in args.question_ids.split(",")]

    # 是否从断点继续
    resume = args.resume and not args.no_resume

    # 从配置文件获取 max_concurrency，命令行参数优先
    multi_eval_config = config.get("multidimensional_evaluation", {})
    max_concurrency = args.max_concurrency if args.max_concurrency is not None else multi_eval_config.get("max_concurrency", 1)

    # 创建评测器
    evaluator = MultidimensionalEvaluator(
        config=config,
        enabled_methods=enabled_methods,
        max_concurrency=max_concurrency
    )

    # 运行评测
    await evaluator.run(
        difficulty=args.difficulty,
        question_ids=question_ids,
        resume=resume,
        filename=args.filename
    )


if __name__ == "__main__":
    asyncio.run(main())

