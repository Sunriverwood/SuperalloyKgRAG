"""
SuperalloyKgRAG 评测模块

提供自动化评测功能，支持：
- 从 JSON 文件加载评测题目
- 调用 GraphRouter 获取回答
- 根据难度级别进行分级评分
- 生成评测报告

模块结构：
- scoring.py: 分级评分器 (L1/L2/L3/L4)
- auto_evaluator.py: 自动评测主模块
- run_evaluation.py: 命令行入口
"""

from .scoring import (
    BaseScorer,
    KeywordMatcher,
    SemanticScorer,
    LLMJudge,
    L1L2Scorer,
    L3Scorer,
    L4Scorer,
    ScorerFactory
)

from .auto_evaluator import (
    EvaluationDataLoader,
    AutoEvaluator,
    run_evaluation
)

__all__ = [
    # 评分器
    "BaseScorer",
    "KeywordMatcher",
    "SemanticScorer",
    "LLMJudge",
    "L1L2Scorer",
    "L3Scorer",
    "L4Scorer",
    "ScorerFactory",
    # 评测器
    "EvaluationDataLoader",
    "AutoEvaluator",
    "run_evaluation"
]

__version__ = "1.0.0"

