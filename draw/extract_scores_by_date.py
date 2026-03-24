"""
从 data/answers 目录下按日期筛选 JSONL 评估文件，
提取每道题在不同模型/方法下的得分，输出为 CSV。

用法示例：
    python draw/extract_scores_by_date.py --date 20260126
    python draw/extract_scores_by_date.py --date 20260126 --output visualizations/scores_20260126.csv
"""
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# 手动映射：evaluation 文件时间戳 -> 方法名
# 格式: "HHMMSS" -> "method_name"
# 可根据 data/reports 下的报告文件对应关系维护
EVALUATION_METHOD_MAP: Dict[str, Dict[str, str]] = {
    "20260126": {
        "161303": "router",
        "175308": "drift",
        "191344": "reasoning",
    },
    "20251229": {
        "110133": "router",
        "112718": "local",
        "114829": "drift",
        "121416": "drift",
        "134231": "drift_llm",
    },
    "20260114": {
        "222043": "reasoning",
    },
}


def _extract_method_name(filepath: Path, date: str = "") -> str:
    """
    根据文件名推断模型/方法名称：
    - baseline_{model}_{date}_{time}.jsonl => {model}
    - evaluation_{date}_{time}_{method}.jsonl => {method}
    - evaluation_{date}_{time}.jsonl => 从 EVALUATION_METHOD_MAP 查找，否则用时间戳
    """
    name = filepath.stem

    # baseline pattern: baseline_{model}_{date}_{time}
    if name.startswith("baseline_"):
        # e.g. baseline_qwen3-max_20260126_163050 -> qwen3-max
        parts = name.split("_")
        # parts: ['baseline', 'qwen3-max', '20260126', '163050']
        # or ['baseline', 'ERNIE-4.5-Turbo-128K', '20260126', '173635']
        # model is parts[1:-2] joined (in case model name has underscores)
        if len(parts) >= 4:
            model = "_".join(parts[1:-2])
            return model
        return name

    # evaluation pattern: evaluation_{date}_{time} or evaluation_{date}_{time}_{method}
    match = re.match(r"evaluation_(\d{8})_(\d{6})(?:_(.+))?", name)
    if match:
        file_date = match.group(1)
        file_time = match.group(2)
        method_suffix = match.group(3)

        # 如果文件名已有方法后缀，直接使用
        if method_suffix:
            return method_suffix

        # 否则从映射表查找
        if file_date in EVALUATION_METHOD_MAP:
            if file_time in EVALUATION_METHOD_MAP[file_date]:
                return EVALUATION_METHOD_MAP[file_date][file_time]

        # 都找不到则返回 "eval_{time}"
        return f"eval_{file_time}"

    return name


def _file_matches_date(filepath: Path, date: str) -> bool:
    """检查文件名是否包含指定日期"""
    return date in filepath.name


def _load_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    """加载 JSONL 文件"""
    records = []
    with filepath.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def _get_score(record: Dict[str, Any]) -> Optional[float]:
    """提取分数"""
    if "overall_score" in record:
        try:
            return float(record["overall_score"])
        except (TypeError, ValueError):
            pass
    scores = record.get("scores")
    if isinstance(scores, dict) and "overall_score" in scores:
        try:
            return float(scores["overall_score"])
        except (TypeError, ValueError):
            pass
    return None


def collect_scores(answers_dir: Path, date: str) -> Tuple[Dict[int, Dict[str, Any]], List[str]]:
    """
    收集指定日期的所有问题得分。
    返回：
    - questions: {qid: {question, difficulty, type, domain, <method>: score, ...}}
    - methods: 所有方法/模型名称列表（保持顺序）
    """
    questions: Dict[int, Dict[str, Any]] = {}
    methods_set: Dict[str, int] = {}  # method -> order
    method_order = 0

    # 1) baseline files
    baseline_dir = answers_dir / "baseline"
    if baseline_dir.exists():
        for fp in sorted(baseline_dir.glob("*.jsonl")):
            if not _file_matches_date(fp, date):
                continue
            method = _extract_method_name(fp, date)
            if method not in methods_set:
                methods_set[method] = method_order
                method_order += 1
            for rec in _load_jsonl(fp):
                qid = rec.get("id")
                if qid is None:
                    continue
                qid = int(qid)
                if qid not in questions:
                    questions[qid] = {
                        "difficulty": rec.get("difficulty", ""),
                        "type": rec.get("type", ""),
                        "domain": rec.get("domain", ""),
                    }
                score = _get_score(rec)
                if score is not None:
                    questions[qid][method] = score

    # 2) evaluation files (non-baseline)
    for fp in sorted(answers_dir.glob("evaluation_*.jsonl")):
        if not _file_matches_date(fp, date):
            continue
        method = _extract_method_name(fp, date)
        if method not in methods_set:
            methods_set[method] = method_order
            method_order += 1
        for rec in _load_jsonl(fp):
            qid = rec.get("id")
            if qid is None:
                continue
            qid = int(qid)
            if qid not in questions:
                questions[qid] = {
                    "difficulty": rec.get("difficulty", ""),
                    "type": rec.get("type", ""),
                    "domain": rec.get("domain", ""),
                }
            score = _get_score(rec)
            if score is not None:
                questions[qid][method] = score

    methods = sorted(methods_set.keys(), key=lambda m: methods_set[m])
    return questions, methods


def write_csv(questions: Dict[int, Dict[str, Any]], methods: List[str], out_path: Path) -> None:
    """写入 CSV 文件"""
    fieldnames = ["question_id", "difficulty", "type", "domain"] + methods
    rows = []
    for qid in sorted(questions.keys()):
        info = questions[qid]
        row = {
            "question_id": qid,
            "difficulty": info.get("difficulty", ""),
            "type": info.get("type", ""),
            "domain": info.get("domain", ""),
        }
        for m in methods:
            row[m] = info.get(m, "")
        rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract per-question scores from JSONL evaluation files by date to CSV."
    )
    parser.add_argument(
        "--date",
        type=str,
        required=True,
        help="Date filter (e.g., 20260126). Only files containing this date will be processed.",
    )
    parser.add_argument(
        "--answers-dir",
        type=Path,
        default=Path("data/answers"),
        help="Directory containing evaluation JSONL files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Default: visualizations/scores_{date}.csv",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = Path(f"visualizations/scores_{args.date}.csv")

    print(f"Processing files for date: {args.date}")
    print(f"Answers directory: {args.answers_dir}")

    questions, methods = collect_scores(args.answers_dir, args.date)

    if not questions:
        print(f"No questions found for date {args.date}")
        return

    write_csv(questions, methods, args.output)
    print(f"\nOutput: {args.output}")
    print(f"Questions: {len(questions)}")
    print(f"Methods/Models: {len(methods)}")
    print("Columns:", methods)


if __name__ == "__main__":
    main()
