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
1. 从 data/evaluation_sets/ 加载评测题目（L12.json / L3.json / L4.json / hard.json）
2. 异步调用 GraphRouter 获取回答
3. 使用分级评分器评估回答质量
4. 将结果保存到 data/answers/（由 settings.yaml evaluation.answers_output_dir 控制）

多维对比与消融实验请使用 evaluation/multidimensional_evaluator.py，
答案按实验子目录写入 data/answers/multidimensional_evaluation/<run_dir>/。
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import yaml
import collections.abc

# --- 项目根目录定义 ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "core" / "query_qwen"))

# 导入评分模块
from evaluation.scoring import ScorerFactory


def load_config(settings_filename: str = "settings.yaml") -> Dict[str, Any]:
    """加载YAML配置文件"""
    config_path = PROJECT_ROOT / "config" / settings_filename
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件 {config_path} 未找到！")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def deep_update(d: dict, u: dict) -> dict:
    """递归深度更新字典"""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def apply_ablation_config(config: Dict[str, Any], ablation_name: str) -> Dict[str, Any]:
    """将消融实验配置深层覆盖到主配置"""
    ablation_profiles = config.get("ablation", {})
    if ablation_name not in ablation_profiles:
        raise ValueError(f"未找到消融实验配置: '{ablation_name}'，可用: {list(ablation_profiles.keys())}")

    profile = ablation_profiles[ablation_name]
    logging.info(f"🔬 应用消融实验配置: '{ablation_name}' - {profile.get('description', '')}")

    for section, overrides in profile.items():
        if section in ['description', 'multidimensional_evaluation']:
            continue
        if section in config and isinstance(config[section], dict) and isinstance(overrides, dict):
            deep_update(config[section], overrides)
            logging.info(f"   ✅ 已深层覆盖 '{section}' 配置")

    return config


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
            question_ids: Optional[List[int]] = None,
            filename: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        加载评测题目

        Args:
            difficulty: 指定难度级别 (L1/L2/L3/L4)，None 表示加载全部
            question_ids: 指定题目 ID 列表，None 表示加载全部
            filename: 指定文件名 (如 'hard.json')，优先级最高

        Returns:
            题目列表
        """
        questions = []

        # 确定要加载的文件
        if filename:
            # 如果指定了文件名，只加载该文件
            files = [filename]
        elif difficulty:
            difficulty = difficulty.upper()
            if difficulty in ["L1", "L2"]:
                files = ["L12.json"]
            elif difficulty == "L3":
                files = ["L3.json"]
            elif difficulty == "L4":
                files = ["L4.json"]
            else:
                files = ["L12.json", "L3.json", "L4.json", "hard.json"]
        else:
            files = ["L12.json", "L3.json", "L4.json", "hard.json"]

        offset = 0

        # 加载文件
        for filename in files:
            filepath = self.data_dir / filename
            if not filepath.exists():
                logging.warning(f"评测文件未找到: {filepath}")
                continue

            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 为每个问题添加来源文件信息和全局 ID
            for i, item in enumerate(data):
                global_qid = offset + i + 1
                item["source_file"] = filename
                item["original_id"] = item.get("id")
                item["id"] = global_qid
                if "difficulty" not in item:
                    difficulty_map = {
                        "L12.json": item.get("difficulty", "L1"),
                        "L3.json": "L3",
                        "L4.json": "L4",
                        "hard.json": "L4",
                    }
                    item["difficulty"] = difficulty_map.get(filename, "L1")

            questions.extend(data)
            logging.info(f"从 {filename} 加载了 {len(data)} 道题目, 全局ID范围 {offset + 1}-{offset + len(data)}")
            offset += len(data)

        # 按难度筛选（针对 L12.json 中混合的 L1/L2）
        if difficulty and difficulty in ["L1", "L2"]:
            questions = [q for q in questions if q.get("difficulty", "").upper() == difficulty]

        # 按 original_id 或 ID 筛选（为向后兼容，如果使用了 question_ids，则匹配 original_id 或全局 id）
        if question_ids:
            questions = [q for q in questions if q.get("original_id") in question_ids or q.get("id") in question_ids]

        logging.info(f"共加载 {len(questions)} 道题目")
        return questions


class AutoEvaluator:
    """
    自动评测器 - 一站式评测解决方案

    功能：
    1. 加载评测题目
    2. 异步调用查询系统获取回答
    3. 多级评分系统评估回答质量
    4. 生成详细评测报告

    使用示例：
        evaluator = AutoEvaluator.from_config()
        report = await evaluator.run(difficulty="L3", max_concurrency=5)
    """

    def __init__(self, config: Dict[str, Any], max_concurrency: int = 5, query_mode: Optional[str] = None):
        """
        初始化评测器

        Args:
            config: 配置字典
            max_concurrency: 最大并发数
            query_mode: 指定查询模式 ('local', 'global', 'reasoning', 'drift', None=自动路由)
        """
        self.config = config
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)

        # 查询模式设置
        self.query_mode = query_mode.lower() if query_mode else None
        if self.query_mode and self.query_mode not in ['basic_rag', 'local', 'global', 'reasoning', 'drift']:
            raise ValueError(f"无效的查询模式: {query_mode}. 可选值: basic_rag, local, global, reasoning, drift")

        # 初始化评分器工厂
        self.scorer_factory = ScorerFactory(config)

        # 初始化路由器（延迟加载）
        self._router = None

        # 初始化具体查询模块（延迟加载）
        self._basic_rag_query = None
        self._local_query = None
        self._global_query = None
        self._reasoning_query = None
        self._drift_query = None

        # 初始化数据加载器
        self.data_loader = EvaluationDataLoader()

        # 输出目录
        self.output_dir = PROJECT_ROOT / "data" / "answers"
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # 报告目录
        self.report_dir = PROJECT_ROOT / "data" / "reports"
        self.report_dir.mkdir(exist_ok=True, parents=True)

        if self.query_mode:
            logging.info(f"评测器将使用指定查询模式: {self.query_mode.upper()}")
        else:
            logging.info("评测器将使用自动路由模式")
            
        # API请求间隔
        self.request_interval = config.get("multidimensional_evaluation", {}).get("request_interval", 1.0)
        
        # 断点控制
        self.checkpoint_file = self.report_dir / "auto_evaluator_checkpoint.json"
        
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
            "last_updated": datetime.now().isoformat()
        }
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_config(cls, settings_filename: str = "settings.yaml", max_concurrency: int = 5, query_mode: Optional[str] = None):
        """
        从配置文件创建评测器实例

        Args:
            settings_filename: 配置文件名
            max_concurrency: 最大并发数
            query_mode: 指定查询模式 ('local', 'global', 'reasoning', 'drift', None=自动路由)

        Returns:
            AutoEvaluator 实例
        """
        config = load_config(settings_filename)
        setup_logging(config)
        return cls(config, max_concurrency, query_mode)

    @property
    def router(self):
        """延迟加载 GraphRouter"""
        if self._router is None:
            from core.query_qwen.router_qwen import GraphRouter
            self._router = GraphRouter(self.config)
            logging.info("GraphRouter 初始化完成")
        return self._router

    @property
    def basic_rag_query(self):
        """延迟加载 RAGQueryHandler"""
        if self._basic_rag_query is None:
            from core.query_qwen.basic_rag_qwen import RAGQueryHandler
            self._basic_rag_query = RAGQueryHandler(self.config)
            logging.info("RAGQueryHandler 初始化完成")
        return self._basic_rag_query

    @property
    def local_query(self):
        """延迟加载 LocalQueryHandler"""
        if self._local_query is None:
            from core.query_qwen.local_query_qwen import LocalQueryHandler
            self._local_query = LocalQueryHandler(self.config)
            logging.info("LocalQueryHandler 初始化完成")
        return self._local_query

    @property
    def global_query(self):
        """延迟加载 GlobalQueryHandler"""
        if self._global_query is None:
            from core.query_qwen.global_query_qwen import GlobalQueryHandler
            self._global_query = GlobalQueryHandler(self.config)
            logging.info("GlobalQueryHandler 初始化完成")
        return self._global_query

    @property
    def reasoning_query(self):
        """延迟加载 ReasoningQueryHandler"""
        if self._reasoning_query is None:
            from core.query_qwen.reasoning_query_qwen import ReasoningQueryHandler
            self._reasoning_query = ReasoningQueryHandler(self.config)
            logging.info("ReasoningQueryHandler 初始化完成")
        return self._reasoning_query

    @property
    def drift_query(self):
        """延迟加载 DriftSearchHandler"""
        if self._drift_query is None:
            from core.query_qwen.router_qwen import DriftSearchHandler
            self._drift_query = DriftSearchHandler(self.config)
            logging.info("DriftSearchHandler 初始化完成")
        return self._drift_query

    async def get_answer(self, question: str) -> Dict[str, Any]:
        """
        调用查询模块获取回答

        Args:
            question: 问题文本

        Returns:
            查询结果字典，包含 answer 和其他模式特有字段
            - answer: 模型回答（始终存在）
            - reasoning_info: 推理模式下的完整推理信息（推理模式时存在）
        """
        async with self.semaphore:
            try:
                result = {}
                if self.query_mode is None:
                    # 使用自动路由
                    answer = await self.router.route_and_answer(question)
                    result['answer'] = answer
                elif self.query_mode == 'basic_rag':
                    answer = self.basic_rag_query.answer_query(question)
                    result['answer'] = answer
                elif self.query_mode == 'local':
                    answer = await self.local_query.answer_query(question)
                    result['answer'] = answer
                elif self.query_mode == 'global':
                    answer = await self.global_query.answer_query(question)
                    result['answer'] = answer
                elif self.query_mode == 'reasoning':
                    # ReasoningQueryHandler.query 是同步方法，使用 to_thread 运行
                    # 返回完整的推理结果（包含 top_nodes, paths, explanations 等）
                    reasoning_result = await asyncio.to_thread(
                        self.reasoning_query.query,
                        question,
                        method='ppr',
                        include_llm_answer=True
                    )
                    result['answer'] = reasoning_result.get('answer', '未能生成推理答案')
                    # 保存完整的推理信息
                    result['reasoning_info'] = {
                        'top_nodes': reasoning_result.get('top_nodes', []),
                        'paths': reasoning_result.get('paths', []),
                        'explanations': reasoning_result.get('explanations', []),
                        'keywords': reasoning_result.get('keywords', []),
                        'num_paths': reasoning_result.get('num_paths', 0),
                    }
                elif self.query_mode == 'drift':
                    answer = await self.drift_query.perform_drift_search(question)
                    result['answer'] = answer
                else:
                    raise ValueError(f"未知的查询模式: {self.query_mode}")

                return result
            except Exception as e:
                logging.error(f"获取回答失败 (模式: {self.query_mode or '自动路由'}): {e}")
                return {'answer': f"[ERROR] {str(e)}"}

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
        source_file = item.get("source_file", "")

        logging.info(f"[{question_id}] 开始评测: {question[:50]}...")

        # 获取模型回答
        start_time = datetime.now()
        query_result = await self.get_answer(question)
        answer_time = (datetime.now() - start_time).total_seconds()
        answer = query_result.get('answer', '[ERROR] 未能获取回答')

        logging.info(f"[{question_id}] 获取回答耗时: {answer_time:.2f}s")

        # 评分
        score_start = datetime.now()
        score_result = self.scorer_factory.score(
            question=question,
            answer=answer,
            ground_truth=ground_truth,
            difficulty=difficulty,
            question_type=question_type,
            domain=domain,
            source_file=source_file
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

        # 如果是推理模式，添加推理信息
        if 'reasoning_info' in query_result:
            result['reasoning_info'] = query_result['reasoning_info']

        return result

    async def evaluate_batch(
            self,
            questions: List[Dict[str, Any]],
            save_intermediate: bool = True,
            resume: bool = False
    ) -> List[Dict[str, Any]]:
        """
        批量评测题目 (并行处理)

        Args:
            questions: 题目列表
            save_intermediate: 是否保存中间结果

        Returns:
            评测结果列表
        """
        total = len(questions)
        
        # 生成输出文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"evaluation_{timestamp}.jsonl"

        # 处理断点续传
        completed_ids = []
        if resume:
            checkpoint = self._load_checkpoint()
            completed_ids = checkpoint.get("completed_ids", [])
            if completed_ids:
                logging.info(f"发现断点记录，已完成 {len(completed_ids)} 道题目，将跳过这些题目。")
            
            # 分离已完成和未完成的题目
            pending_questions = [q for q in questions if q.get("id") not in completed_ids]
            
            # 如果存在中间结果文件，读取已有的结果
            if completed_ids and output_file.exists():
                with open(output_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            res = json.loads(line)
                            if res.get("id") in completed_ids:
                                results.append(res)
        else:
            pending_questions = questions
            
        total_pending = len(pending_questions)
        if total_pending == 0:
            logging.info("所有题目都已评测完成。")
            return results

        logging.info(f"开始批量评测，待评测 {total_pending} 道题目 (并发限制: {self.max_concurrency})")
        logging.info(f"结果将保存到: {output_file}")

        async def _eval_with_delay(idx: int, item: Dict[str, Any]) -> Dict[str, Any]:
            if idx > 0 and self.request_interval > 0:
                await asyncio.sleep(self.request_interval)
            return await self.evaluate_single(item)

        # 创建任务列表
        tasks = [_eval_with_delay(idx, item) for idx, item in enumerate(pending_questions)]
        
        # 使用 as_completed 实时获取完成的任务，以便显示进度和保存中间结果
        for i, completed_task in enumerate(asyncio.as_completed(tasks), 1):
            try:
                result = await completed_task
                results.append(result)
                completed_ids.append(result.get("id"))

                # 保存中间结果
                if save_intermediate:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    # 同步保存断点
                    self._save_checkpoint(completed_ids)

                logging.info(f"进度: {i}/{total_pending} ({100 * i / total_pending:.1f}%) - ID: {result.get('id')}")

            except Exception as e:
                logging.error(f"评测任务执行失败: {e}")

        # 如果全部完成则清理断点
        if len(results) == total:
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()

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

    async def run(
            self,
            difficulty: Optional[str] = None,
            question_ids: Optional[List[int]] = None,
            save_intermediate: bool = True,
            filename: Optional[str] = None,
            resume: bool = False
    ) -> Dict[str, Any]:
        """
        运行完整评测流程的主入口方法

        Args:
            difficulty: 指定难度级别 (L1/L2/L3/L4)，None 表示全部
            question_ids: 指定题目 ID 列表，None 表示全部
            save_intermediate: 是否保存中间结果
            filename: 指定评测文件名 (如 'hard.json')

        Returns:
            评测报告字典
        """
        logging.info("=" * 80)
        logging.info("开始自动评测流程")
        logging.info("=" * 80)

        # 加载题目
        questions = self.data_loader.load_questions(
            difficulty=difficulty,
            question_ids=question_ids,
            filename=filename
        )

        if not questions:
            logging.warning("没有找到符合条件的题目")
            return {"error": "No questions found"}

        # 执行评测
        results = await self.evaluate_batch(questions, save_intermediate=save_intermediate, resume=resume)

        # 生成报告
        report = self.generate_report(results)

        logging.info("=" * 80)
        logging.info("评测流程完成")
        logging.info("=" * 80)

        return report


if __name__ == "__main__":
    # 命令行接口
    import argparse

    parser = argparse.ArgumentParser(description="自动评测系统")
    parser.add_argument("--difficulty", type=str, default=None, help="难度级别 (L1/L2/L3/L4)")
    parser.add_argument("--ids", type=str, default=None, help="题目ID列表，逗号分隔")
    parser.add_argument("--concurrency", type=int, default=5, help="最大并发数")
    parser.add_argument("--settings", type=str, default="settings.yaml", help="配置文件名")
    parser.add_argument("--mode", type=str, default=None,
                       choices=['basic_rag', 'local', 'global', 'reasoning', 'drift'],
                       help="指定查询模式 (basic_rag/local/global/reasoning/drift)，不指定则使用自动路由")
    parser.add_argument("--filename", type=str, default=None, help="指定评测文件名 (如 hard.json)")
    parser.add_argument("--ablation", type=str, default=None, help="使用指定的消融实验配置名称")
    parser.add_argument("--resume", action="store_true", help="启用断点续传")

    args = parser.parse_args()

    # 解析题目 ID
    question_ids = None
    if args.ids:
        question_ids = [int(x.strip()) for x in args.ids.split(",")]

    # 加载底层配置并可能覆盖消融参数
    config = load_config(args.settings)
    if args.ablation:
        config = apply_ablation_config(config, args.ablation)

    # 创建评测器并运行
    # 初始化时不通过文件由于我们需要注入覆盖后的配置，所以直接调用类并重设日志
    setup_logging(config)
    evaluator = AutoEvaluator(
        config=config,
        max_concurrency=args.concurrency,
        query_mode=args.mode
    )

    report = asyncio.run(evaluator.run(
        difficulty=args.difficulty,
        question_ids=question_ids,
        filename=args.filename,
        resume=args.resume
    ))

    print("\n" + "=" * 80)
    print("评测报告摘要")
    print("=" * 80)
    print(json.dumps(report, ensure_ascii=False, indent=2))




