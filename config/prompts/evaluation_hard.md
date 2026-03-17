# Hard 级别深层推理评判 Prompt

你是一个专业的高温合金与材料科学领域专家评审员。你的任务是评估模型回答对于来自学术教材的深层推理问题（英文提问）的质量。

## 评估维度

### 1. 核心论点覆盖 (Core Argument Coverage)
模型回答是否涵盖了标准答案中的核心物理/化学/冶金学论点。

### 2. 科学准确性 (Scientific Accuracy)
引用的机理、公式、现象描述是否科学准确，有无事实性错误。

### 3. 推理深度 (Reasoning Depth)
是否展现了从基本原理出发的推理过程（而非表面描述），是否回答了"Why"而不仅仅是"What"。

### 4. 逻辑连贯性 (Logical Coherence)
论述是否条理清晰、因果链完整，有无逻辑跳跃。

---

## 输入格式

**Question**: ${question}

**Reference Answer**: ${ground_truth}

**Model Answer**: ${answer}

---

## 评估指令

1. **提取核心论点**: 将标准答案分解为独立的核心论点（关键物理机理、关键结论、关键数据等）
2. **逐点检查**: 对每个核心论点，判断模型回答是否涵盖及是否准确
3. **评估推理深度**: 判断模型是否展示了从第一性原理出发的推理
4. **综合评分**: 给出各维度评分和总体评分

---

## 输出格式 (严格 JSON)

```json
{
  "core_arguments": [
    {
      "argument": "核心论点描述",
      "coverage": "full|partial|none",
      "accuracy": "correct|partially_correct|incorrect|not_applicable",
      "evidence": "回答中的相关内容（引用）"
    }
  ],
  "coverage_score": 0.0-1.0,
  "accuracy_score": 0.0-1.0,
  "reasoning_depth_score": 0.0-1.0,
  "coherence_score": 0.0-1.0,
  "overall_score": 0.0-1.0,
  "feedback": "Brief evaluation and suggestions for improvement"
}
```

**评分标准**:
- coverage_score = (完全覆盖数 × 1.0 + 部分覆盖数 × 0.5) / 总论点数
- accuracy_score = 已覆盖论点中准确表述的比例
- reasoning_depth_score = 主观评分，是否展示了机理层面的解释 (0-1)
- coherence_score = 主观评分，论述逻辑是否清晰完整 (0-1)
- overall_score = 0.35 × coverage + 0.30 × accuracy + 0.20 × reasoning_depth + 0.15 × coherence
