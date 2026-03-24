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
重新评分模块 - 利用 multidimensional_evaluator 已有答案，使用 auto_evaluator 的评分体系重新评分

用法:
    # 对所有方法重新评分
    python -m evaluation.rescore

    # 只对指定方法重新评分
    python -m evaluation.rescore --methods "basic_rag,global,local,reasoning,router"

    # 指定答案目录时间戳
    python -m evaluation.rescore --timestamp 20260314_133442
"""

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.auto_evaluator import EvaluationDataLoader
from evaluation.scoring import ScorerFactory


def load_config(settings_filename: str = "settings.yaml") -> Dict[str, Any]:
    config_path = PROJECT_ROOT / "config" / settings_filename
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_logging(config: Dict[str, Any]):
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
    log_file = PROJECT_ROOT / "logs" / "rescore.log"
    log_file.parent.mkdir(exist_ok=True, parents=True)

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


def load_answers(answers_dir: Path, method: str, timestamp: str) -> Dict[int, Dict[str, Any]]:
    """
    加载指定方法的答案文件，返回 {global_question_id: {question, answer, latency}} 字典。

    答案文件中的 question_id 在每个难度组内各自从 1 开始（非全局唯一），
    因此使用行序位置（1-based）作为全局唯一 ID。
    """
    answer_file = answers_dir / f"{method}_answers_{timestamp}.jsonl"
    if not answer_file.exists():
        logging.warning(f"答案文件不存在: {answer_file}")
        return {}

    answers = {}
    with open(answer_file, 'r', encoding='utf-8') as f:
        for global_qid, line in enumerate(f, start=1):
            if line.strip():
                record = json.loads(line)
                record["original_question_id"] = record["question_id"]
                record["question_id"] = global_qid
                answers[global_qid] = record
    logging.info(f"从 {answer_file.name} 加载了 {len(answers)} 条答案")
    return answers


def build_question_index(questions: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """构建 question_id -> question_info 的索引"""
    return {q["id"]: q for q in questions}


def build_global_question_index(data_loader: 'EvaluationDataLoader') -> Dict[int, Dict[str, Any]]:
    """
    根据题目文件的加载顺序构建 全局question_id -> question_info 的索引。

    答案文件中的 question_id 是全局递增的（跨所有难度等级），但各题目文件的 id 均从 1 开始。
    此函数按文件加载顺序（L12, L3, L4, hard）累计偏移量，将全局 question_id 映射到正确的题目。
    """
    files_order = ["L12.json", "L3.json", "L4.json", "hard.json"]
    global_index = {}
    offset = 0

    for filename in files_order:
        filepath = data_loader.data_dir / filename
        if not filepath.exists():
            logging.warning(f"题目文件未找到: {filepath}")
            continue

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for i, item in enumerate(data):
            global_qid = offset + i + 1
            item["source_file"] = filename
            # 如果题目本身没有 difficulty 字段，则根据文件名推断
            if "difficulty" not in item:
                difficulty_map = {
                    "L12.json": item.get("difficulty", "L1"),
                    "L3.json": "L3",
                    "L4.json": "L4",
                    "hard.json": "L4",
                }
                item["difficulty"] = difficulty_map.get(filename, "L1")
            global_index[global_qid] = item

        logging.info(f"文件 {filename}: {len(data)} 道题, 全局ID范围 {offset + 1}-{offset + len(data)}")
        offset += len(data)

    logging.info(f"全局题目索引共 {len(global_index)} 道题 (全局ID: 1-{offset})")
    return global_index


def rescore(
        config: Dict[str, Any],
        methods: Optional[List[str]] = None,
        timestamp: Optional[str] = None,
        max_workers: int = 3,
        answers_dir_path: Optional[str] = None
):
    """主评分流程"""
    base_answers_dir = PROJECT_ROOT / "data" / "answers" / "multidimensional_evaluation"
    base_reports_dir = PROJECT_ROOT / "data" / "reports" / "rescore"

    if answers_dir_path:
        answers_dir = base_answers_dir / answers_dir_path
        output_dir = base_reports_dir / answers_dir_path
    else:
        answers_dir = base_answers_dir
        output_dir = base_reports_dir

    output_dir.mkdir(exist_ok=True, parents=True)

    # 自动检测时间戳
    if not timestamp:
        answer_files = list(answers_dir.glob("*_answers_*.jsonl"))
        if not answer_files:
            logging.error(f"在 {answers_dir} 中未找到答案文件")
            return
        # 提取最新时间戳
        timestamps = set()
        for f in answer_files:
            parts = f.stem.split("_answers_")
            if len(parts) == 2:
                timestamps.add(parts[1])
        timestamp = sorted(timestamps)[-1]
        logging.info(f"自动检测到时间戳: {timestamp}")

    # 自动检测方法列表
    if not methods:
        answer_files = list(answers_dir.glob(f"*_answers_{timestamp}.jsonl"))
        methods = []
        for f in answer_files:
            method_name = f.stem.replace(f"_answers_{timestamp}", "")
            methods.append(method_name)
        logging.info(f"自动检测到 {len(methods)} 个方法: {methods}")

    # 加载评测题目（包含 ground_truth 和 difficulty）
    # 使用全局索引：按文件加载顺序累计偏移，将全局 question_id 映射到正确的题目
    data_loader = EvaluationDataLoader()
    question_index = build_global_question_index(data_loader)

    # 初始化评分器
    scorer_factory = ScorerFactory(config)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 逐方法评分
    all_reports = {}
    for method in methods:
        logging.info(f"\n{'='*60}")
        logging.info(f"开始对方法 [{method}] 重新评分...")
        logging.info(f"{'='*60}")

        answers = load_answers(answers_dir, method, timestamp)
        if not answers:
            continue

        # 准备评分任务
        tasks = []  # (qid, q_info, answer_record)
        error_results = []
        for qid, answer_record in sorted(answers.items()):
            q_info = question_index.get(qid)
            if not q_info:
                logging.warning(f"[Q{qid}] 在题目库中未找到对应题目，跳过")
                continue
            answer_text = answer_record["answer"]
            if answer_text.startswith("[ERROR]"):
                logging.warning(f"[Q{qid}] 方法 {method} 回答为错误，跳过评分")
                error_results.append({
                    "question_id": qid, "question": q_info["question"],
                    "difficulty": q_info.get("difficulty", "L1"), "method": method,
                    "answer": answer_text, "scores": {"error": True, "overall_score": 0.0},
                    "overall_score": 0.0
                })
                continue
            tasks.append((qid, q_info, answer_record))

        def _score_one(qid, q_info, answer_record):
            """单题评分函数（线程池中执行）"""
            question = q_info["question"]
            ground_truth = q_info.get("ground_truth", "")
            difficulty = q_info.get("difficulty", "L1")
            question_type = q_info.get("type", "")
            domain = q_info.get("domain", "")
            source_file = q_info.get("source_file", "")
            answer_text = answer_record["answer"]

            logging.info(f"[Q{qid}] 评分中... (difficulty={difficulty}, source={source_file})")
            try:
                score_result = scorer_factory.score(
                    question=question, answer=answer_text, ground_truth=ground_truth,
                    difficulty=difficulty, question_type=question_type,
                    domain=domain, source_file=source_file
                )
            except Exception as e:
                logging.error(f"[Q{qid}] 评分失败: {e}")
                score_result = {"error": str(e), "overall_score": 0.0}

            overall = score_result.get("overall_score", 0)
            logging.info(f"[Q{qid}] 得分: {overall:.3f} ({score_result.get('scoring_method', 'unknown')})")
            return {
                "question_id": qid, "question": question,
                "difficulty": difficulty, "type": question_type,
                "domain": domain, "source_file": source_file,
                "method": method, "answer": answer_text,
                "latency_seconds": answer_record.get("latency_seconds", 0),
                "scores": score_result, "overall_score": overall
            }

        # 并发评分
        results = list(error_results)
        completed = 0
        total = len(tasks)
        logging.info(f"方法 [{method}] 共 {total} 道题待评分，并发数: {max_workers}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(_score_one, qid, q_info, ar): qid
                for qid, q_info, ar in tasks
            }
            for future in as_completed(future_map):
                completed += 1
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    qid = future_map[future]
                    logging.error(f"[Q{qid}] 线程异常: {e}")
                if completed % 10 == 0 or completed == total:
                    logging.info(f"方法 [{method}] 进度: {completed}/{total}")

        # 保存该方法的评分结果
        result_file = output_dir / f"{method}_scores_{run_timestamp}.jsonl"
        with open(result_file, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        logging.info(f"方法 [{method}] 评分完成，{len(results)} 条结果保存到 {result_file.name}")

        # 统计该方法的分数
        valid_scores = [r["overall_score"] for r in results if not r.get("scores", {}).get("error")]
        if valid_scores:
            avg = sum(valid_scores) / len(valid_scores)
            all_reports[method] = {
                "avg_score": round(avg, 4),
                "total": len(results),
                "valid": len(valid_scores),
                "min": round(min(valid_scores), 4),
                "max": round(max(valid_scores), 4)
            }
            logging.info(f"方法 [{method}] 平均分: {avg:.4f} (n={len(valid_scores)})")

    # 生成汇总报告
    summary = {
        "report_type": "rescore_summary",
        "source_timestamp": timestamp,
        "rescore_timestamp": run_timestamp,
        "methods": all_reports
    }
    summary_file = output_dir / f"rescore_summary_{run_timestamp}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 打印汇总
    print(f"\n{'='*70}")
    print(f"重新评分汇总 (source: {timestamp})")
    print(f"{'='*70}")
    print(f"{'方法':<35} {'平均分':>8} {'有效数':>6} {'最低分':>8} {'最高分':>8}")
    print(f"{'-'*70}")
    for method, stats in sorted(all_reports.items(), key=lambda x: x[1]["avg_score"], reverse=True):
        print(f"{method:<35} {stats['avg_score']:>8.4f} {stats['valid']:>6} {stats['min']:>8.4f} {stats['max']:>8.4f}")
    print(f"{'='*70}")
    print(f"汇总报告: {summary_file}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="利用已有答案重新评分")
    parser.add_argument("--methods", type=str, default=None,
                        help="指定方法（逗号分隔），如: basic_rag,global,local")
    parser.add_argument("--timestamp", type=str, default=None,
                        help="答案文件时间戳，如: 20260314_133442（不指定则自动检测最新）")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="并发评分线程数（默认5，建议3-5避免API限流）")
    parser.add_argument("--settings", type=str, default="settings.yaml",
                        help="配置文件名")
    parser.add_argument("--answers_dir", type=str, default=None,
                        help="指定答案所在的子目录名称，例如：ablation_text_only")
    args = parser.parse_args()

    config = load_config(args.settings)
    setup_logging(config)

    methods = [m.strip() for m in args.methods.split(",")] if args.methods else None

    rescore(
        config, 
        methods=methods, 
        timestamp=args.timestamp, 
        max_workers=args.concurrency,
        answers_dir_path=args.answers_dir
    )
