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
SuperalloyKgRAG 评测模块

提供自动化评测功能，支持：
- 从 JSON 文件加载评测题目
- 调用 GraphRouter 获取回答
- 根据难度级别进行分级评分
- 生成评测报告

模块结构：
- scoring.py: 分级评分器 (L1/L2/L3/L4)
- auto_evaluator.py: 自动评测主模块（已整合 run_evaluation 功能）
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
    AutoEvaluator
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
    "AutoEvaluator"
]

__version__ = "1.0.0"

