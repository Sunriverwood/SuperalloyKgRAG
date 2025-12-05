"""
评分模块 - 针对不同难度级别的回答质量评估

评分策略：
- L1/L2: 关键词/数值匹配 + 语义相似度
- L3: 要点覆盖率 + LLM-as-Judge
- L4: 推理链完整性评估 (LLM-as-Judge)
"""

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from string import Template
from typing import Dict, Any, List

import numpy as np
from openai import OpenAI

# --- 项目根目录定义 ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class BaseScorer(ABC):
    """评分器基类"""

    @abstractmethod
    def score(self, question: str, answer: str, ground_truth: str, **kwargs) -> Dict[str, Any]:
        """
        评估回答质量

        Args:
            question: 问题
            answer: 模型回答
            ground_truth: 标准答案
            **kwargs: 额外参数

        Returns:
            包含评分和详细信息的字典
        """
        pass


class KeywordMatcher:
    """关键词/数值匹配器 - 用于 L1/L2 事实性问题"""

    # 数值模式：匹配整数、小数、百分比、范围等
    NUMBER_PATTERN = re.compile(
        r'(?:约|大约|接近|超过|低于|小于|大于)?'
        r'[\d]+(?:\.[\d]+)?'
        r'(?:\s*[-~至到]\s*[\d]+(?:\.[\d]+)?)?'
        r'(?:\s*[%％℃°CKGhμn²³]|MPa|GPa|mm|nm|wt\.?%|at\.?%|vol\.?%)?',
        re.IGNORECASE
    )

    # 化学式/合金名称模式
    ALLOY_PATTERN = re.compile(
        r'(?:Ni|Co|Fe|Cr|Mo|W|Re|Ru|Ta|Ti|Al|Nb|V|Hf|Zr|B|C|Y|La|Ce)'
        r'[\d]*(?:[.\d]*)?'
        r'(?:\s*[-_]?\s*(?:Ni|Co|Fe|Cr|Mo|W|Re|Ru|Ta|Ti|Al|Nb|V|Hf|Zr|B|C|Y|La|Ce)[\d]*(?:[.\d]*)?)*'
        r'|(?:IN|CMSX|RR|TMS|PWA|René|Waspaloy|Inconel|Hastelloy|Mar-M|Udimet|Nimonic)'
        r'[\s\-]?[\d]+[A-Za-z]*',
        re.IGNORECASE
    )

    # 专业术语列表（可扩展）
    TECHNICAL_TERMS = [
        # 相与结构
        "γ相", "γ'相", "γ\"相", "gamma", "gamma prime", "δ相", "σ相", "μ相", "P相",
        "TCP相", "拓扑密排相", "FCC", "BCC", "HCP", "L12", "D022",
        # 强化机制
        "固溶强化", "沉淀强化", "弥散强化", "晶界强化", "奥罗万", "Orowan",
        "位错", "滑移", "攀移", "交滑移", "层错", "反相畴界", "APB",
        # 工艺
        "定向凝固", "单晶", "真空熔炼", "VIM", "VAR", "ESR", "HIP",
        "粉末冶金", "铸造", "锻造", "热处理", "时效", "固溶处理",
        # 性能
        "蠕变", "疲劳", "氧化", "腐蚀", "屈服", "断裂", "韧性", "硬度",
        "筏化", "rafting", "蠕变断裂", "低周疲劳", "高周疲劳",
    ]

    def extract_keywords(self, text: str) -> Dict[str, List[str]]:
        """从文本中提取关键信息"""
        keywords = {
            "numbers": [],
            "alloys": [],
            "terms": []
        }

        # 提取数值
        numbers = self.NUMBER_PATTERN.findall(text)
        keywords["numbers"] = [n.strip() for n in numbers if n.strip()]

        # 提取合金名称
        alloys = self.ALLOY_PATTERN.findall(text)
        keywords["alloys"] = list(set([a.strip() for a in alloys if a.strip()]))

        # 提取专业术语
        text_lower = text.lower()
        for term in self.TECHNICAL_TERMS:
            if term.lower() in text_lower:
                keywords["terms"].append(term)

        return keywords

    def calculate_match_score(
            self,
            answer_keywords: Dict[str, List[str]],
            truth_keywords: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """计算关键词匹配得分"""
        scores = {}

        for key in ["numbers", "alloys", "terms"]:
            answer_set = set([str(k).lower() for k in answer_keywords.get(key, [])])
            truth_set = set([str(k).lower() for k in truth_keywords.get(key, [])])

            if not truth_set:
                scores[f"{key}_precision"] = 1.0
                scores[f"{key}_recall"] = 1.0
                scores[f"{key}_f1"] = 1.0
                continue

            if not answer_set:
                scores[f"{key}_precision"] = 0.0
                scores[f"{key}_recall"] = 0.0
                scores[f"{key}_f1"] = 0.0
                continue

            # 计算交集（考虑部分匹配）
            matched = 0
            for t in truth_set:
                for a in answer_set:
                    if t in a or a in t:
                        matched += 1
                        break

            precision = matched / len(answer_set) if answer_set else 0
            recall = matched / len(truth_set) if truth_set else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            scores[f"{key}_precision"] = precision
            scores[f"{key}_recall"] = recall
            scores[f"{key}_f1"] = f1

        # 综合得分
        f1_scores = [scores.get(f"{k}_f1", 0) for k in ["numbers", "alloys", "terms"]]
        # 数值和术语权重更高
        weights = [0.4, 0.2, 0.4]
        scores["weighted_f1"] = sum(s * w for s, w in zip(f1_scores, weights))

        return scores


class SemanticScorer:
    """语义相似度评分器 - 使用 Embedding 计算余弦相似度"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = os.environ.get("QWEN_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self.embedding_model = config.get("embedding", {}).get("model", "text-embedding-v4")
        self.dimensionality = config.get("embedding", {}).get("dimensionality", 768)

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def get_embedding(self, text: str) -> np.ndarray:
        """获取文本的 embedding 向量"""
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text[:8000],  # 截断过长文本
                dimensions=self.dimensionality
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            logging.error(f"获取 embedding 失败: {e}")
            return np.zeros(self.dimensionality)

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def score(self, answer: str, ground_truth: str) -> float:
        """计算语义相似度得分"""
        answer_vec = self.get_embedding(answer)
        truth_vec = self.get_embedding(ground_truth)
        return self.cosine_similarity(answer_vec, truth_vec)


class LLMJudge:
    """LLM-as-Judge 评判器 - 用于 L3/L4 复杂评估"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = os.environ.get("QWEN_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self.model_name = config.get("query", {}).get("generation_model", "qwen3-max")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        # 加载评判 Prompt 模板
        self.prompts = {}
        prompt_dir = PROJECT_ROOT / "config" / "prompts"
        for level in ["l3", "l4"]:
            prompt_path = prompt_dir / f"evaluation_{level}.md"
            if prompt_path.exists():
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    self.prompts[level] = f.read()
            else:
                logging.warning(f"评判 Prompt 未找到: {prompt_path}")

    def judge(
            self,
            question: str,
            answer: str,
            ground_truth: str,
            level: str = "l3"
    ) -> Dict[str, Any]:
        """
        使用 LLM 进行评判

        Args:
            question: 问题
            answer: 模型回答
            ground_truth: 标准答案
            level: 难度级别 (l3/l4)

        Returns:
            评判结果字典
        """
        if level not in self.prompts:
            logging.error(f"未找到 {level} 级别的评判 Prompt")
            return {"error": f"Missing prompt for level {level}"}

        # 使用 Template 替换变量
        template = Template(self.prompts[level])
        prompt = template.safe_substitute(
            question=question,
            ground_truth=ground_truth,
            answer=answer
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1  # 低温度以保证评判一致性
            )
            result_text = response.choices[0].message.content

            # 解析 JSON 结果
            json_match = re.search(r'\{[\s\S]*}', result_text)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                logging.warning("LLM 评判结果未包含有效 JSON")
                return {
                    "raw_response": result_text,
                    "overall_score": 0.5,
                    "error": "Failed to parse JSON"
                }

        except json.JSONDecodeError as e:
            logging.error(f"JSON 解析失败: {e}")
            return {"error": str(e), "overall_score": 0.0}
        except Exception as e:
            logging.error(f"LLM 评判失败: {e}")
            return {"error": str(e), "overall_score": 0.0}


class L1L2Scorer(BaseScorer):
    """L1/L2 级别评分器 - 事实检索和简单推理"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.keyword_matcher = KeywordMatcher()
        self.semantic_scorer = SemanticScorer(config)

    def score(self, question: str, answer: str, ground_truth: str, **kwargs) -> Dict[str, Any]:
        """
        L1/L2 评分：关键词匹配 + 语义相似度

        评分权重：
        - 关键词匹配 F1: 60%
        - 语义相似度: 40%
        """
        # 关键词匹配
        answer_keywords = self.keyword_matcher.extract_keywords(answer)
        truth_keywords = self.keyword_matcher.extract_keywords(ground_truth)
        keyword_scores = self.keyword_matcher.calculate_match_score(answer_keywords, truth_keywords)

        # 语义相似度
        semantic_score = self.semantic_scorer.score(answer, ground_truth)

        # 综合得分
        keyword_f1 = keyword_scores.get("weighted_f1", 0)
        overall_score = 0.6 * keyword_f1 + 0.4 * semantic_score

        return {
            "overall_score": overall_score,
            "keyword_scores": keyword_scores,
            "semantic_score": semantic_score,
            "extracted_keywords": {
                "answer": answer_keywords,
                "ground_truth": truth_keywords
            },
            "difficulty": kwargs.get("difficulty", "L1/L2"),
            "scoring_method": "keyword_matching + semantic_similarity"
        }


class L3Scorer(BaseScorer):
    """L3 级别评分器 - 综合性问题"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm_judge = LLMJudge(config)
        self.semantic_scorer = SemanticScorer(config)

    def score(self, question: str, answer: str, ground_truth: str, **kwargs) -> Dict[str, Any]:
        """
        L3 评分：LLM-as-Judge 要点覆盖率评估

        评分维度：
        - 要点覆盖率 (coverage_score)
        - 准确性 (accuracy_score)
        - 逻辑性 (coherence_score)
        """
        # LLM 评判
        llm_result = self.llm_judge.judge(question, answer, ground_truth, level="l3")

        # 语义相似度作为补充
        semantic_score = self.semantic_scorer.score(answer, ground_truth)

        # 如果 LLM 评判失败，使用语义相似度作为备选
        if "error" in llm_result and llm_result.get("overall_score", 0) == 0:
            overall_score = semantic_score
        else:
            # LLM 评判结果权重更高
            llm_score = llm_result.get("overall_score", 0.5)
            overall_score = 0.8 * llm_score + 0.2 * semantic_score

        return {
            "overall_score": overall_score,
            "llm_judgment": llm_result,
            "semantic_score": semantic_score,
            "coverage_score": llm_result.get("coverage_score", 0),
            "accuracy_score": llm_result.get("accuracy_score", 0),
            "coherence_score": llm_result.get("coherence_score", 0),
            "key_points": llm_result.get("key_points", []),
            "feedback": llm_result.get("feedback", ""),
            "difficulty": "L3",
            "scoring_method": "llm_judge + semantic_similarity"
        }


class L4Scorer(BaseScorer):
    """L4 级别评分器 - 设计/发现任务"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm_judge = LLMJudge(config)

    def score(self, question: str, answer: str, ground_truth: str, **kwargs) -> Dict[str, Any]:
        """
        L4 评分：推理链完整性评估

        评分维度：
        - 冲突识别 (conflict_identification)
        - 机理调用 (mechanism_invocation)
        - 方案给出 (solution_proposal)
        """
        # LLM 评判
        llm_result = self.llm_judge.judge(question, answer, ground_truth, level="l4")

        # 提取各维度得分
        conflict_score = llm_result.get("conflict_identification", {}).get("score", 0)
        mechanism_score = llm_result.get("mechanism_invocation", {}).get("score", 0)
        solution_score = llm_result.get("solution_proposal", {}).get("score", 0)

        # 计算综合得分（如果 LLM 没有返回 overall_score）
        if "overall_score" not in llm_result or llm_result.get("error"):
            overall_score = 0.3 * conflict_score + 0.35 * mechanism_score + 0.35 * solution_score
        else:
            overall_score = llm_result["overall_score"]

        return {
            "overall_score": overall_score,
            "conflict_identification": llm_result.get("conflict_identification", {}),
            "mechanism_invocation": llm_result.get("mechanism_invocation", {}),
            "solution_proposal": llm_result.get("solution_proposal", {}),
            "reasoning_chain_complete": llm_result.get("reasoning_chain_complete", False),
            "feedback": llm_result.get("feedback", ""),
            "difficulty": "L4",
            "scoring_method": "llm_judge_reasoning_chain"
        }


class ScorerFactory:
    """评分器工厂 - 根据难度级别创建对应的评分器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._scorers = {}

    def get_scorer(self, difficulty: str) -> BaseScorer:
        """
        根据难度级别获取评分器

        Args:
            difficulty: 难度级别 (L1/L2/L3/L4)

        Returns:
            对应的评分器实例
        """
        difficulty = difficulty.upper()

        if difficulty not in self._scorers:
            if difficulty in ["L1", "L2"]:
                self._scorers[difficulty] = L1L2Scorer(self.config)
            elif difficulty == "L3":
                self._scorers[difficulty] = L3Scorer(self.config)
            elif difficulty == "L4":
                self._scorers[difficulty] = L4Scorer(self.config)
            else:
                logging.warning(f"未知难度级别 {difficulty}，使用 L1L2Scorer")
                self._scorers[difficulty] = L1L2Scorer(self.config)

        return self._scorers[difficulty]

    def score(
            self,
            question: str,
            answer: str,
            ground_truth: str,
            difficulty: str,
            **kwargs
    ) -> Dict[str, Any]:
        """
        评分入口函数

        Args:
            question: 问题
            answer: 模型回答
            ground_truth: 标准答案
            difficulty: 难度级别
            **kwargs: 额外参数

        Returns:
            评分结果字典
        """
        scorer = self.get_scorer(difficulty)
        return scorer.score(question, answer, ground_truth, difficulty=difficulty, **kwargs)

