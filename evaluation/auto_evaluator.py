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
自动评测主模块 - 加载题目、调用查询模块、保存结果

功能：
1. 从 data/evaluation_sets/*.json 加载评测题目
2. 异步调用 GraphRouter 获取回答
3. 使用分级评分器评估回答质量
4. 将结果保存到 data/answers/
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import yaml

# --- 项目根目录定义 ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "core" / "query_qwen"))

# 导入评分模块
from scoring import ScorerFactory


def load_config(settings_filename: str = "settings.yaml") -> Dict[str, Any]:
    """加载YAML配置文件"""
    config_path = PROJECT_ROOT / "config" / settings_filename
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件 {config_path} 未找到！")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_logging(config: Dict[str, Any], log_name: str = "evaluation"):
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
    logging.info(f"评测日志记录器设置完成: {log_file}")
    logging.info("=" * 80)


class EvaluationDataLoader:
    """评测数据加载器"""

    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or (PROJECT_ROOT / "data" / "evaluation_sets")

    def load_questions(
            self,
            difficulty: Optional[str] = None,
            question_ids: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """
        加载评测题目

        Args:
            difficulty: 指定难度级别 (L1/L2/L3/L4)，None 表示加载全部
            question_ids: 指定题目 ID 列表，None 表示加载全部

        Returns:
            题目列表
        """
        questions = []

        # 确定要加载的文件
        if difficulty:
            difficulty = difficulty.upper()
            if difficulty in ["L1", "L2"]:
                files = ["L12.json"]
            elif difficulty == "L3":
                files = ["L3.json"]
            elif difficulty == "L4":
                files = ["L4.json"]
            else:
                files = ["L12.json", "L3.json", "L4.json"]
        else:
            files = ["L12.json", "L3.json", "L4.json"]

        # 加载文件
        for filename in files:
            filepath = self.data_dir / filename
            if not filepath.exists():
                logging.warning(f"评测文件未找到: {filepath}")
                continue

            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 为每个问题添加来源文件信息
            for item in data:
                item["source_file"] = filename

            questions.extend(data)
            logging.info(f"从 {filename} 加载了 {len(data)} 道题目")

        # 按难度筛选（针对 L12.json 中混合的 L1/L2）
        if difficulty and difficulty in ["L1", "L2"]:
            questions = [q for q in questions if q.get("difficulty", "").upper() == difficulty]

        # 按 ID 筛选
        if question_ids:
            questions = [q for q in questions if q.get("id") in question_ids]

        logging.info(f"共加载 {len(questions)} 道题目")
        return questions


class AutoEvaluator:
    """自动评测器"""

    def __init__(self, config: Dict[str, Any], max_concurrency: int = 5):
        """
        初始化评测器

        Args:
            config: 配置字典
            max_concurrency: 最大并发数
        """
        self.config = config
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)

        # 初始化评分器工厂
        self.scorer_factory = ScorerFactory(config)

        # 初始化路由器（延迟加载）
        self._router = None

        # 输出目录
        self.output_dir = PROJECT_ROOT / "data" / "answers"
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # 报告目录
        self.report_dir = PROJECT_ROOT / "data" / "reports"
        self.report_dir.mkdir(exist_ok=True, parents=True)

    @property
    def router(self):
        """延迟加载 GraphRouter"""
        if self._router is None:
            from core.query_qwen.router_qwen import GraphRouter
            self._router = GraphRouter(self.config)
            logging.info("GraphRouter 初始化完成")
        return self._router

    async def get_answer(self, question: str) -> str:
        """
        调用 GraphRouter 获取回答

        Args:
            question: 问题文本

        Returns:
            模型回答
        """
        async with self.semaphore:
            try:
                answer = await self.router.route_and_answer(question)
                return answer
            except Exception as e:
                logging.error(f"获取回答失败: {e}")
                return f"[ERROR] {str(e)}"

    async def evaluate_single(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        评测单道题目

        Args:
            item: 题目字典

        Returns:
            评测结果
        """
        question_id = item.get("id")
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")
        difficulty = item.get("difficulty", "L1")
        question_type = item.get("type", "")
        domain = item.get("domain", "")
        theme = item.get("theme", "")

        logging.info(f"[{question_id}] 开始评测: {question[:50]}...")

        # 获取模型回答
        start_time = datetime.now()
        answer = await self.get_answer(question)
        answer_time = (datetime.now() - start_time).total_seconds()

        logging.info(f"[{question_id}] 获取回答耗时: {answer_time:.2f}s")

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

        logging.info(f"[{question_id}] 评分完成: {score_result.get('overall_score', 0):.3f}, 耗时: {score_time:.2f}s")

        # 构建结果
        result = {
            "id": question_id,
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

    async def evaluate_batch(
            self,
            questions: List[Dict[str, Any]],
            save_intermediate: bool = True
    ) -> List[Dict[str, Any]]:
        """
        批量评测题目

        Args:
            questions: 题目列表
            save_intermediate: 是否保存中间结果

        Returns:
            评测结果列表
        """
        results = []
        total = len(questions)

        # 生成输出文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"evaluation_{timestamp}.jsonl"

        logging.info(f"开始批量评测，共 {total} 道题目")
        logging.info(f"结果将保存到: {output_file}")

        for i, item in enumerate(questions, 1):
            try:
                result = await self.evaluate_single(item)
                results.append(result)

                # 保存中间结果
                if save_intermediate:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')

                logging.info(f"进度: {i}/{total} ({100 * i / total:.1f}%)")

            except Exception as e:
                logging.error(f"评测题目 {item.get('id')} 失败: {e}")
                error_result = {
                    "id": item.get("id"),
                    "question": item.get("question"),
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                results.append(error_result)

        logging.info(f"批量评测完成，共 {len(results)} 条结果")
        return results

    def generate_report(
            self,
            results: List[Dict[str, Any]],
            report_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        生成评测报告

        Args:
            results: 评测结果列表
            report_name: 报告名称

        Returns:
            报告字典
        """
        if not results:
            return {"error": "No results to report"}

        # 按难度分组统计
        difficulty_stats = {}
        domain_stats = {}
        type_stats = {}

        for r in results:
            if "error" in r:
                continue

            difficulty = r.get("difficulty", "Unknown")
            domain = r.get("domain", "Unknown")
            q_type = r.get("type", "Unknown")
            score = r.get("overall_score", 0)

            # 难度统计
            if difficulty not in difficulty_stats:
                difficulty_stats[difficulty] = {"scores": [], "count": 0}
            difficulty_stats[difficulty]["scores"].append(score)
            difficulty_stats[difficulty]["count"] += 1

            # 领域统计
            if domain not in domain_stats:
                domain_stats[domain] = {"scores": [], "count": 0}
            domain_stats[domain]["scores"].append(score)
            domain_stats[domain]["count"] += 1

            # 类型统计
            if q_type not in type_stats:
                type_stats[q_type] = {"scores": [], "count": 0}
            type_stats[q_type]["scores"].append(score)
            type_stats[q_type]["count"] += 1

        # 计算平均分
        def calc_avg(stats_dict):
            for key, val in stats_dict.items():
                scores = val["scores"]
                val["avg_score"] = sum(scores) / len(scores) if scores else 0
                val["min_score"] = min(scores) if scores else 0
                val["max_score"] = max(scores) if scores else 0
                del val["scores"]  # 移除原始分数列表
            return stats_dict

        difficulty_stats = calc_avg(difficulty_stats)
        domain_stats = calc_avg(domain_stats)
        type_stats = calc_avg(type_stats)

        # 总体统计
        all_scores = [r.get("overall_score", 0) for r in results if "error" not in r]
        overall_stats = {
            "total_questions": len(results),
            "successful_evaluations": len(all_scores),
            "failed_evaluations": len(results) - len(all_scores),
            "avg_score": sum(all_scores) / len(all_scores) if all_scores else 0,
            "min_score": min(all_scores) if all_scores else 0,
            "max_score": max(all_scores) if all_scores else 0
        }

        report = {
            "report_name": report_name or f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "generated_at": datetime.now().isoformat(),
            "overall_statistics": overall_stats,
            "by_difficulty": difficulty_stats,
            "by_domain": domain_stats,
            "by_type": type_stats
        }

        # 保存报告
        report_file = self.report_dir / f"{report['report_name']}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logging.info(f"评测报告已保存: {report_file}")

        return report


async def run_evaluation(
        difficulty: Optional[str] = None,
        question_ids: Optional[List[int]] = None,
        max_concurrency: int = 5,
        save_intermediate: bool = True
) -> Dict[str, Any]:
    """
    运行评测的入口函数

    Args:
        difficulty: 指定难度级别
        question_ids: 指定题目 ID
        max_concurrency: 最大并发数
        save_intermediate: 是否保存中间结果

    Returns:
        评测报告
    """
    # 加载配置
    config = load_config()
    setup_logging(config)

    # 加载题目
    loader = EvaluationDataLoader()
    questions = loader.load_questions(difficulty=difficulty, question_ids=question_ids)

    if not questions:
        logging.warning("没有找到符合条件的题目")
        return {"error": "No questions found"}

    # 初始化评测器
    evaluator = AutoEvaluator(config, max_concurrency=max_concurrency)

    # 执行评测
    results = await evaluator.evaluate_batch(questions, save_intermediate=save_intermediate)

    # 生成报告
    report = evaluator.generate_report(results)

    return report


if __name__ == "__main__":
    # 简单测试
    import argparse

    parser = argparse.ArgumentParser(description="自动评测系统")
    parser.add_argument("--difficulty", type=str, default=None, help="难度级别 (L1/L2/L3/L4)")
    parser.add_argument("--ids", type=str, default=None, help="题目ID列表，逗号分隔")
    parser.add_argument("--concurrency", type=int, default=5, help="最大并发数")

    args = parser.parse_args()

    # 解析题目 ID
    question_ids = None
    if args.ids:
        question_ids = [int(x.strip()) for x in args.ids.split(",")]

    # 运行评测
    report = asyncio.run(run_evaluation(
        difficulty=args.difficulty,
        question_ids=question_ids,
        max_concurrency=args.concurrency
    ))

    print("\n" + "=" * 80)
    print("评测报告摘要")
    print("=" * 80)
    print(json.dumps(report, ensure_ascii=False, indent=2))

