#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评测系统命令行入口

用法示例：
    # 评测所有题目
    python run_evaluation.py

    # 评测指定难度
    python run_evaluation.py --difficulty L3

    # 评测指定题目
    python run_evaluation.py --ids 1,2,3,4,5

    # 指定并发数
    python run_evaluation.py --difficulty L4 --concurrency 3

    # 指定输出目录
    python run_evaluation.py --output ./my_results/
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# --- 项目根目录定义 ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from auto_evaluator import (
    load_config,
    setup_logging,
    EvaluationDataLoader,
    AutoEvaluator,
    run_evaluation
)


def print_banner():
    """打印启动横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     ███████╗██╗   ██╗ █████╗ ██╗     ██╗   ██╗ █████╗ ████████╗██╗ ██████╗ ║
║     ██╔════╝██║   ██║██╔══██╗██║     ██║   ██║██╔══██╗╚══██╔══╝██║██╔═══██╗║
║     █████╗  ██║   ██║███████║██║     ██║   ██║███████║   ██║   ██║██║   ██║║
║     ██╔══╝  ╚██╗ ██╔╝██╔══██║██║     ██║   ██║██╔══██║   ██║   ██║██║   ██║║
║     ███████╗ ╚████╔╝ ██║  ██║███████╗╚██████╔╝██║  ██║   ██║   ██║╚██████╔╝║
║     ╚══════╝  ╚═══╝  ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝║
║                                                                              ║
║                 SuperalloyKgRAG 自动评测系统 v1.0                            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_summary(report: dict):
    """打印评测结果摘要"""
    print("\n" + "═" * 80)
    print("                           📊 评测结果摘要")
    print("═" * 80)

    overall = report.get("overall_statistics", {})
    print(f"\n📌 总体统计:")
    print(f"   • 评测题目数: {overall.get('total_questions', 0)}")
    print(f"   • 成功评测数: {overall.get('successful_evaluations', 0)}")
    print(f"   • 失败评测数: {overall.get('failed_evaluations', 0)}")
    print(f"   • 平均得分: {overall.get('avg_score', 0):.4f}")
    print(f"   • 最高得分: {overall.get('max_score', 0):.4f}")
    print(f"   • 最低得分: {overall.get('min_score', 0):.4f}")

    # 按难度统计
    by_difficulty = report.get("by_difficulty", {})
    if by_difficulty:
        print(f"\n📌 按难度统计:")
        for diff, stats in sorted(by_difficulty.items()):
            print(f"   • {diff}: 平均 {stats.get('avg_score', 0):.4f} "
                  f"(共 {stats.get('count', 0)} 题, "
                  f"范围 {stats.get('min_score', 0):.4f} - {stats.get('max_score', 0):.4f})")

    # 按领域统计
    by_domain = report.get("by_domain", {})
    if by_domain:
        print(f"\n📌 按领域统计:")
        for domain, stats in sorted(by_domain.items()):
            print(f"   • {domain}: 平均 {stats.get('avg_score', 0):.4f} "
                  f"(共 {stats.get('count', 0)} 题)")

    # 按类型统计
    by_type = report.get("by_type", {})
    if by_type:
        print(f"\n📌 按类型统计:")
        for qtype, stats in sorted(by_type.items()):
            print(f"   • {qtype}: 平均 {stats.get('avg_score', 0):.4f} "
                  f"(共 {stats.get('count', 0)} 题)")

    print("\n" + "═" * 80)


def list_questions(difficulty: str = None):
    """列出可用的评测题目"""
    loader = EvaluationDataLoader()
    questions = loader.load_questions(difficulty=difficulty)

    print(f"\n📋 可用题目列表 (共 {len(questions)} 道):")
    print("-" * 80)
    print(f"{'ID':<5} {'难度':<5} {'类型':<20} {'领域':<15} {'问题':<40}")
    print("-" * 80)

    for q in questions:
        question_text = q.get("question", "")[:37] + "..." if len(q.get("question", "")) > 40 else q.get("question", "")
        print(f"{q.get('id', 'N/A'):<5} {q.get('difficulty', 'N/A'):<5} "
              f"{q.get('type', 'N/A'):<20} {q.get('domain', 'N/A'):<15} "
              f"{question_text:<40}")

    print("-" * 80)


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="SuperalloyKgRAG 自动评测系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python run_evaluation.py                      # 评测所有题目
  python run_evaluation.py --difficulty L3      # 仅评测 L3 难度
  python run_evaluation.py --ids 1,2,3          # 评测指定 ID 的题目
  python run_evaluation.py --list               # 列出所有可用题目
  python run_evaluation.py --list --difficulty L4  # 列出 L4 难度题目
        """
    )

    parser.add_argument(
        "--difficulty", "-d",
        type=str,
        choices=["L1", "L2", "L3", "L4", "l1", "l2", "l3", "l4"],
        default=None,
        help="指定难度级别 (L1/L2/L3/L4)"
    )

    parser.add_argument(
        "--ids", "-i",
        type=str,
        default=None,
        help="指定题目 ID 列表，逗号分隔 (例如: 1,2,3,4,5)"
    )

    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=5,
        help="最大并发数 (默认: 5)"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="自定义输出目录"
    )

    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="列出可用的评测题目"
    )

    parser.add_argument(
        "--no-intermediate",
        action="store_true",
        help="不保存中间结果"
    )

    parser.add_argument(
        "--report-only",
        type=str,
        default=None,
        help="仅从已有结果文件生成报告 (提供 JSONL 文件路径)"
    )

    args = parser.parse_args()

    # 打印横幅
    print_banner()

    # 如果仅列出题目
    if args.list:
        list_questions(args.difficulty)
        return

    # 如果仅生成报告
    if args.report_only:
        config = load_config()
        setup_logging(config)

        results = []
        with open(args.report_only, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))

        evaluator = AutoEvaluator(config)
        report = evaluator.generate_report(results)
        print_summary(report)
        return

    # 解析题目 ID
    question_ids = None
    if args.ids:
        try:
            question_ids = [int(x.strip()) for x in args.ids.split(",")]
        except ValueError:
            print("❌ 错误: --ids 参数格式不正确，请使用逗号分隔的数字")
            sys.exit(1)

    # 标准化难度参数
    difficulty = args.difficulty.upper() if args.difficulty else None

    print(f"\n⚙️  评测配置:")
    print(f"   • 难度级别: {difficulty or '全部'}")
    print(f"   • 题目 ID: {question_ids or '全部'}")
    print(f"   • 最大并发: {args.concurrency}")
    print(f"   • 保存中间结果: {'否' if args.no_intermediate else '是'}")

    # 确认开始
    print(f"\n🚀 开始评测...\n")
    start_time = datetime.now()

    try:
        # 运行评测
        report = await run_evaluation(
            difficulty=difficulty,
            question_ids=question_ids,
            max_concurrency=args.concurrency,
            save_intermediate=not args.no_intermediate
        )

        # 计算耗时
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\n⏱️  总耗时: {elapsed:.2f} 秒")

        # 打印摘要
        print_summary(report)

        # 如果有错误
        if "error" in report:
            print(f"\n⚠️  评测过程中出现错误: {report['error']}")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  评测被用户中断")
        sys.exit(130)
    except Exception as e:
        logging.exception(f"评测过程中出现异常: {e}")
        print(f"\n❌ 评测失败: {e}")
        sys.exit(1)

    print("\n✅ 评测完成!")


if __name__ == "__main__":
    asyncio.run(main())

