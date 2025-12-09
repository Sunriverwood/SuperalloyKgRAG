# 严格模式使用指南

## 概述

严格模式 (Strict Mode) 是推理查询系统的一个重要功能，用于控制 LLM 在生成答案时是否仅使用知识图谱数据。

## 快速开始

### 命令行使用

```bash
# 严格模式（默认）- 仅使用知识图谱数据
python core/query_qwen/reasoning_query_qwen.py --query "镍的用途是什么？"

# 非严格模式 - 允许使用 LLM 自身知识
python core/query_qwen/reasoning_query_qwen.py --query "镍的用途是什么？" --no-strict
```

### 交互模式

```bash
python core/query_qwen/reasoning_query_qwen.py --interactive
```

系统会询问：
```
Use strict mode (LLM only uses knowledge graph data)? (yes/no) [yes]:
```

- 回答 `yes` 或直接回车：启用严格模式
- 回答 `no`：禁用严格模式

### Python 代码

```python
from core.query_qwen.reasoning_query_qwen import ReasoningQueryHandler, load_config

config = load_config()
handler = ReasoningQueryHandler(config)

# 严格模式
results = handler.query(
    query_text="镍的用途是什么？",
    strict_mode=True  # 默认值
)

# 非严格模式
results = handler.query(
    query_text="镍的用途是什么？",
    strict_mode=False
)
```

## 两种模式对比

| 特性 | 严格模式 (strict_mode=True) | 非严格模式 (strict_mode=False) |
|------|---------------------------|------------------------------|
| 数据来源 | 仅知识图谱 | 知识图谱 + LLM 通用知识 |
| 答案可追溯性 | ✅ 高 | ⚠️ 中等 |
| 避免幻觉 | ✅ 强 | ⚠️ 一般 |
| 答案完整性 | ⚠️ 取决于知识图谱 | ✅ 高 |
| 适用场景 | 需要确保答案基于特定数据源 | 需要更全面的答案 |

## 严格模式工作原理

### 系统提示词

严格模式使用特殊的系统提示词来约束 LLM：

```
You are a knowledge graph reasoning assistant. 

CRITICAL CONSTRAINTS:
1. You MUST ONLY use information from the provided knowledge graph reasoning results
2. You MUST NOT use your own training data or general knowledge
3. If the provided reasoning results don't contain enough information to answer, 
   say "Based on the available knowledge graph data, I cannot find sufficient 
   information to answer this question."
4. Every statement in your answer must be traceable to the provided entities 
   and reasoning paths
5. Do not make assumptions or inferences beyond what is explicitly stated 
   in the reasoning results
```

### 用户提示词

```
Question: {query}

Knowledge Graph Reasoning Results:
{reasoning_results}

IMPORTANT: Answer ONLY based on the above reasoning results. 
Do NOT use any external knowledge or your training data. 
If the reasoning results are insufficient, explicitly state that.
```

## 最佳实践

### 何时使用严格模式 ✅

1. **企业知识库查询**：确保答案完全基于公司内部数据
2. **科研文献分析**：保证引用的准确性和可追溯性
3. **专业领域查询**：避免通用知识干扰专业判断
4. **审计和合规**：需要明确的数据来源追踪

### 何时使用非严格模式 ⚠️

1. **探索性研究**：需要更广泛的背景信息
2. **教育场景**：可以接受通用知识的补充说明
3. **知识图谱不完整**：已知数据源可能缺少关键信息
4. **概念解释**：需要基础概念的通用解释

## 示例对比

### 查询："镍在超合金中的作用是什么？"

#### 严格模式输出示例

```
Based on the knowledge graph reasoning results:

According to the entity "Nickel" (relevance: 0.95) and its relationship 
with "Superalloy", nickel serves as a primary alloying element in 
superalloys. The reasoning path shows:

Nickel → USED_IN → Superalloy (confidence: 0.89)
Nickel → PROVIDES → Corrosion Resistance (confidence: 0.85)

The knowledge graph indicates that nickel provides corrosion resistance 
and high-temperature stability to superalloys.
```

特点：
- 明确引用推理结果
- 标注相关性分数
- 可追溯到具体实体和关系

#### 非严格模式输出示例

```
Based on the knowledge graph and general metallurgical principles:

Nickel plays a crucial role in superalloys as identified in the reasoning 
results. It acts as a primary alloying element providing:

1. Corrosion resistance (as shown in the knowledge graph)
2. High-temperature stability (confirmed by reasoning paths)
3. Phase stabilization (general knowledge about FCC structure)
4. Oxidation resistance at elevated temperatures

Nickel typically comprises 40-70% of superalloy composition and is 
essential for maintaining the austenitic structure...
```

特点：
- 结合知识图谱和通用知识
- 提供更详细的技术背景
- 可能包含知识图谱外的信息

## 调试和验证

### 检查模式状态

查看日志文件 `logs/superalloyKgRAG.log`：

```
Processing query: 镍的用途是什么？
Mode: method=ppr, include_llm=True, strict_mode=True
```

### 验证答案来源

严格模式的答案应该：
- 包含对推理结果的明确引用
- 提及实体名称和相关性分数
- 说明推理路径

如果答案包含无法追溯的信息，可能：
1. LLM 未严格遵守约束
2. 需要调整提示词
3. 考虑使用温度参数为 0

## 配置建议

### 推荐默认设置

```yaml
query:
  temperature: 0.1  # 降低温度以增强确定性
  strict_mode_default: true  # 默认启用严格模式
```

### 调整 LLM 参数

对于严格模式，建议：
- 降低 `temperature`（0.0-0.3）
- 使用更强的约束提示词
- 启用详细日志记录

## 常见问题

### Q: 严格模式下为什么有时答案是"无法找到足够信息"？

A: 这是正常的。严格模式会诚实地报告知识图谱中没有足够的数据来回答问题，这比编造答案更可靠。

### Q: 如何让严格模式提供更详细的答案？

A: 可以：
1. 增强知识图谱的完整性
2. 调整推理路径的数量
3. 修改提示词以要求更详细的解释

### Q: 严格模式能100%保证不使用外部知识吗？

A: 提示词可以大幅降低使用外部知识的可能性，但 LLM 的行为难以完全控制。建议：
- 使用低温度参数
- 定期审查答案质量
- 结合人工验证

## 技术细节

### 参数传递链

```
命令行参数 --no-strict
    ↓
args.no_strict
    ↓
handler.query(strict_mode=not args.no_strict)
    ↓
generate_answer(strict_mode=strict_mode)
    ↓
system_prompt (根据 strict_mode 选择)
```

### 相关文件

- `core/query_qwen/reasoning_query_qwen.py`: 主实现文件
- `config/settings.yaml`: 配置文件
- `logs/superalloyKgRAG.log`: 运行日志

## 更新历史

- **2025-12-05**: 添加严格模式功能
  - 新增 `strict_mode` 参数
  - 新增 `--no-strict` 命令行标志
  - 增强系统提示词控制

## 反馈和贡献

如果发现严格模式的问题或有改进建议，请：
1. 查看日志文件分析问题
2. 收集问题案例
3. 提交改进建议

---

**建议**: 对于生产环境，始终使用严格模式 (strict_mode=True) 以确保答案的可追溯性和可靠性。

