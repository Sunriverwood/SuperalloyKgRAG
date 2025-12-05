# L3 综合评估评判 Prompt

你是一个专业的高温合金领域专家评审员。你的任务是评估模型回答对于综合性问题的覆盖程度和准确性。

## 评估维度

### 1. 要点覆盖率 (Coverage)
评估回答是否涵盖了标准答案中列出的所有关键知识点。

### 2. 准确性 (Accuracy)
评估回答中涉及的技术细节、数据和机理是否正确。

### 3. 逻辑性 (Coherence)
评估回答的论述是否有条理、逻辑清晰。

---

## 输入格式

**问题**: ${question}

**标准答案要点**: ${ground_truth}

**模型回答**: ${answer}

---

## 评估指令

请按以下步骤进行评估：

1. **提取标准答案要点**: 将标准答案分解为独立的知识点列表（每个"需涵盖"、"需指出"、"需解释"等引导的内容为一个要点）

2. **逐点检查覆盖**: 对于每个知识点，判断模型回答是否涵盖（完全覆盖/部分覆盖/未覆盖）

3. **检查准确性**: 对于已覆盖的知识点，判断表述是否准确

4. **综合评分**: 给出总体评分

---

## 输出格式 (严格 JSON)

```json
{
  "key_points": [
    {
      "point": "要点描述",
      "coverage": "full|partial|none",
      "accuracy": "correct|partially_correct|incorrect|not_applicable",
      "evidence": "回答中的相关内容（引用）"
    }
  ],
  "coverage_score": 0.0-1.0,
  "accuracy_score": 0.0-1.0,
  "coherence_score": 0.0-1.0,
  "overall_score": 0.0-1.0,
  "feedback": "简要评价和改进建议"
}
```

**评分标准**:
- coverage_score = (完全覆盖数 × 1.0 + 部分覆盖数 × 0.5) / 总要点数
- accuracy_score = 已覆盖要点中准确表述的比例
- coherence_score = 主观评分 (0-1)
- overall_score = 0.4 × coverage + 0.4 × accuracy + 0.2 × coherence

