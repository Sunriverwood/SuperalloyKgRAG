# L4 设计/发现任务评判 Prompt

你是一个专业的高温合金领域专家评审员。你的任务是评估模型回答对于设计/发现类问题的推理链完整性和方案合理性。

## 评估维度

L4 级别的设计/发现问题要求回答包含完整的三步推理链：

### 1. 冲突识别 (Conflict Identification)
- 是否正确识别了问题中涉及的核心矛盾/权衡
- 是否理解了不同设计目标之间的冲突

### 2. 机理调用 (Mechanism Invocation)
- 是否正确运用了相关的物理/化学/冶金学机理
- 机理解释是否专业、准确

### 3. 方案给出 (Solution Proposal)
- 是否给出了具体、可操作的设计方案
- 方案是否具有合理性和可行性
- 是否考虑了权衡和副作用

---

## 输入格式

**问题**: ${question}

**标准设计思路**: ${ground_truth}

**模型回答**: ${answer}

---

## 评估指令

请按以下步骤进行评估：

1. **检查冲突识别**: 模型是否识别并阐述了核心设计矛盾？

2. **检查机理调用**: 模型是否正确引用了相关的科学/工程机理？

3. **检查方案质量**: 模型给出的方案是否具体、合理、考虑了权衡？

4. **综合评分**: 对三个维度分别评分并给出总体评价

---

## 输出格式 (严格 JSON)

```json
{
  "conflict_identification": {
    "score": 0.0-1.0,
    "identified_conflicts": ["识别出的冲突1", "识别出的冲突2"],
    "expected_conflicts": ["预期的冲突1", "预期的冲突2"],
    "analysis": "评价说明"
  },
  "mechanism_invocation": {
    "score": 0.0-1.0,
    "invoked_mechanisms": ["引用的机理1", "引用的机理2"],
    "expected_mechanisms": ["预期的机理1", "预期的机理2"],
    "accuracy": "correct|partially_correct|incorrect",
    "analysis": "评价说明"
  },
  "solution_proposal": {
    "score": 0.0-1.0,
    "proposed_solutions": ["方案1", "方案2"],
    "specificity": "high|medium|low",
    "feasibility": "high|medium|low",
    "tradeoff_consideration": true|false,
    "analysis": "评价说明"
  },
  "overall_score": 0.0-1.0,
  "reasoning_chain_complete": true|false,
  "feedback": "综合评价和改进建议"
}
```

**评分标准**:
- conflict_identification.score: 完全识别=1.0, 部分识别=0.5, 未识别=0.0
- mechanism_invocation.score: 正确完整=1.0, 部分正确=0.3-0.7, 错误=0.0
- solution_proposal.score: 具体可行且考虑权衡=1.0, 一般=0.5, 模糊/不可行=0.0
- overall_score = 0.3 × conflict + 0.35 × mechanism + 0.35 × solution
- reasoning_chain_complete: 三个维度得分均 >= 0.5 时为 true

