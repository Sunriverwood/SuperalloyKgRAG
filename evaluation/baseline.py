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
基线评测模块 - 直接调用不同 LLM 模型回答问题并进行评价

功能：
1. 从配置文件加载多个待测试模型的 API 信息
2. 直接将问题发送给模型（不使用 RAG 或特定 Prompt）
3. 使用统一的评分标准对不同模型的回答进行评估
4. 生成对比报告
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import yaml
from openai import OpenAI

# --- 项目根目录定义 ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# 导入现有评测组件
from evaluation.auto_evaluator import EvaluationDataLoader, load_config, setup_logging
from evaluation.scoring import ScorerFactory


class BaselineEvaluator:
    """基线评测器"""

    def __init__(self, config: Dict[str, Any], max_concurrency: int = 5):
        """
        初始化基线评测器

        Args:
            config: 配置字典
            max_concurrency: 最大并发数
        """
        self.config = config
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)

        # 初始化评分器工厂
        self.scorer_factory = ScorerFactory(config)

        # 初始化数据加载器
        self.data_loader = EvaluationDataLoader()

        # 输出目录
        self.output_dir = PROJECT_ROOT / "data" / "answers" / "baseline"
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # 报告目录
        self.report_dir = PROJECT_ROOT / "data" / "reports" / "baseline"
        self.report_dir.mkdir(exist_ok=True, parents=True)

    def _needs_proxy(self, model_name: str) -> bool:
        """判断模型是否需要代理"""
        model_lower = model_name.lower()
        # Gemini 系列或 ChatGPT/GPT 系列需要代理
        return "gemini" in model_lower or "gpt" in model_lower or "chatgpt" in model_lower

    def _resolve_api_key(self, api_key_str: str) -> str:
        """
        解析 API Key，支持 ${VAR} 格式的环境变量引用

        Args:
            api_key_str: 原始 API Key 字符串

        Returns:
            解析后的 API Key
        """
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

    async def get_answer(self, question: str, model_config: Dict[str, Any]) -> str:
        """
        直接调用模型获取回答

        Args:
            question: 问题文本
            model_config: 模型配置信息

        Returns:
            模型回答文本
        """
        async with self.semaphore:
            try:
                api_key = self._resolve_api_key(model_config.get("api_key", ""))
                base_url = model_config.get("base_url")
                model_name = model_config.get("name")

                # 判断是否需要代理（Gemini 或 ChatGPT 系列）
                if self._needs_proxy(model_name):
                    proxy = self.config.get("proxy")
                    if proxy:
                        os.environ["HTTP_PROXY"] = proxy
                        os.environ["HTTPS_PROXY"] = proxy
                        logging.info(f"模型 {model_name} 使用代理: {proxy}")
                else:
                    # 不需要代理时清除代理环境变量
                    os.environ.pop("HTTP_PROXY", None)
                    os.environ.pop("HTTPS_PROXY", None)
                    logging.info(f"模型 {model_name} 不使用代理")

                # 初始化 OpenAI 客户端 (兼容大多数模型提供商)
                client = OpenAI(
                    api_key=api_key,
                    base_url=base_url
                )

                # 直接发送问题，不添加任何系统提示词或 RAG 上下文
                logging.debug(f"正在请求模型 {model_name}...")
                
                # 使用 to_thread 运行同步的 OpenAI 调用
                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=model_name,
                    messages=[
                        {"role": "user", "content": question}
                    ],
                    temperature=model_config.get("temperature", 0.1),
                    max_tokens=model_config.get("max_tokens", 2000)
                )

                answer = response.choices[0].message.content
                return answer.strip()

            except Exception as e:
                logging.error(f"模型 {model_config.get('name')} 获取回答失败: {e}")
                return f"[ERROR] {str(e)}"

    async def evaluate_single(self, item: Dict[str, Any], model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        评测单道题目

        Args:
            item: 题目字典
            model_config: 模型配置

        Returns:
            评测结果字典
        """
        question_id = item.get("id")
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")
        difficulty = item.get("difficulty", "L1")
        question_type = item.get("type", "")
        domain = item.get("domain", "")
        theme = item.get("theme", "")

        logging.info(f"[{model_config['name']}][{question_id}] 开始评测: {question[:50]}...")

        # 获取模型回答
        start_time = datetime.now()
        answer = await self.get_answer(question, model_config)
        answer_time = (datetime.now() - start_time).total_seconds()

        logging.info(f"[{model_config['name']}][{question_id}] 获取回答耗时: {answer_time:.2f}s")

        # 评分
        score_start = datetime.now()
        score_result = self.scorer_factory.score(
            question=question,
            answer=answer,
            ground_truth=ground_truth,
            difficulty=difficulty,
            question_type=question_type,
            domain=domain
        )
        score_time = (datetime.now() - score_start).total_seconds()

        logging.info(f"[{model_config['name']}][{question_id}] 评分完成: {score_result.get('overall_score', 0):.3f}")

        # 构建结果
        result = {
            "id": question_id,
            "model": model_config['name'],
            "question": question,
            "theme": theme,
            "difficulty": difficulty,
            "type": question_type,
            "domain": domain,
            "ground_truth": ground_truth,
            "answer": answer,
            "scores": score_result,
            "overall_score": score_result.get("overall_score", 0),
            "answer_time_seconds": answer_time,
            "score_time_seconds": score_time,
            "timestamp": datetime.now().isoformat()
        }

        return result

    async def run_for_model(
            self, 
            model_config: Dict[str, Any], 
            questions: List[Dict[str, Any]],
            save_intermediate: bool = True
    ) -> List[Dict[str, Any]]:
        """
        为特定模型运行完整评测 (并行处理)

        Args:
            model_config: 模型配置
            questions: 题目列表
            save_intermediate: 是否保存中间结果

        Returns:
            该模型的评测结果列表
        """
        model_name = model_config['name']
        logging.info("=" * 80)
        logging.info(f"开始模型 {model_name} 的基线评测 (并发限制: {self.max_concurrency})")
        logging.info("=" * 80)

        # 生成输出文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"baseline_{model_name}_{timestamp}.jsonl"

        # 创建任务列表
        tasks = [self.evaluate_single(item, model_config) for item in questions]
        
        results = []
        total = len(questions)
        
        # 使用 as_completed 实时获取完成的任务
        for i, completed_task in enumerate(asyncio.as_completed(tasks), 1):
            try:
                result = await completed_task
                results.append(result)

                if save_intermediate:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')

                logging.info(f"[{model_name}] 进度: {i}/{total} ({100 * i / total:.1f}%) - ID: {result.get('id')}")

            except Exception as e:
                logging.error(f"[{model_name}] 评测任务执行失败: {e}")
                
        return results

    def generate_comparison_report(self, all_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        生成多模型对比报告

        Args:
            all_results: 模型名称到结果列表的映射

        Returns:
            对比报告字典
        """
        report = {
            "report_type": "baseline_comparison",
            "generated_at": datetime.now().isoformat(),
            "summary": {},
            "details": {}
        }

        for model_name, results in all_results.items():
            valid_scores = [r["overall_score"] for r in results if "scores" in r]
            if not valid_scores:
                continue
                
            avg_score = sum(valid_scores) / len(valid_scores)
            
            # 按难度统计
            diff_stats = {}
            for r in results:
                if "scores" not in r: continue
                d = r["difficulty"]
                if d not in diff_stats: diff_stats[d] = []
                diff_stats[d].append(r["overall_score"])
            
            diff_avg = {d: sum(s)/len(s) for d, s in diff_stats.items()}

            report["summary"][model_name] = {
                "avg_score": avg_score,
                "total_questions": len(results),
                "by_difficulty": diff_avg
            }
            report["details"][model_name] = results

        # 保存报告
        report_file = self.report_dir / f"baseline_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logging.info(f"对比报告已保存: {report_file}")
        return report

    async def run(
            self, 
            difficulty: Optional[str] = None, 
            question_ids: Optional[List[int]] = None,
            model_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        运行完整基线评测流程的主入口

        Args:
            difficulty: 难度过滤
            question_ids: ID 过滤
            model_names: 模型名称过滤 (None 表示全部)

        Returns:
            对比报告
        """
        # 1. 加载题目
        questions = self.data_loader.load_questions(
            difficulty=difficulty,
            question_ids=question_ids
        )

        if not questions:
            logging.warning("没有找到符合条件的题目")
            return {"error": "No questions found"}

        # 2. 获取待测试模型列表
        models = self.config.get("baseline", {}).get("models", [])
        if not models:
            logging.error("配置文件中未找到 baseline.models 配置")
            return {"error": "No baseline models configured"}

        # 过滤模型
        if model_names:
            models = [m for m in models if m['name'] in model_names]
            if not models:
                logging.error(f"未找到指定的模型: {model_names}")
                return {"error": f"Models {model_names} not found in config"}

        # 3. 依次对每个模型进行评测
        all_results = {}
        for model_config in models:
            results = await self.run_for_model(model_config, questions)
            all_results[model_config['name']] = results

        # 4. 生成对比报告
        report = self.generate_comparison_report(all_results)
        
        logging.info("=" * 80)
        logging.info("所有模型的基线评测已完成")
        logging.info("=" * 80)
        
        return report


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="基线模型对比评测 system")
    parser.add_argument("--difficulty", type=str, default=None, help="难度级别 (L1/L2/L3/L4)")
    parser.add_argument("--ids", type=str, default=None, help="题目ID列表，逗号分隔")
    parser.add_argument("--models", type=str, default=None, help="模型名称列表，逗号分隔")
    parser.add_argument("--concurrency", type=int, default=5, help="最大并发数")
    parser.add_argument("--settings", type=str, default="settings.yaml", help="配置文件名")

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.settings)
    setup_logging(config)

    # 解析题目 ID
    question_ids = None
    if args.ids:
        question_ids = [int(x.strip()) for x in args.ids.split(",")]

    # 解析模型名称
    model_names = None
    if args.models:
        model_names = [x.strip() for x in args.models.split(",")]

    # 创建评测器并运行
    evaluator = BaselineEvaluator(config, max_concurrency=args.concurrency)
    
    report = await evaluator.run(
        difficulty=args.difficulty,
        question_ids=question_ids,
        model_names=model_names
    )

    # 打印摘要
    print("\n" + "=" * 80)
    print("基线评测摘要")
    print("=" * 80)
    for model, stats in report.get("summary", {}).items():
        print(f"模型: {model:<20} | 平均分: {stats['avg_score']:.4f} | 题目数: {stats['total_questions']}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
