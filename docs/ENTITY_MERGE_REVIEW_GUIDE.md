# 实体合并人工审核功能指南

## 概述

本文档介绍实体合并阶段的两个新功能：
1. **批量请求数量控制**：在实体消歧与合并阶段，单次批量请求控制数量最多为 `embedding_batch_size`
2. **人工审核功能**：实体合并阶段增加人工审核功能，可对比大模型合并效果与人工合并效果

## 功能1：批量请求数量控制

### 功能说明

在实体消歧和实体合并阶段，当待处理的实体数量超过配置的 `embedding_batch_size` 时，系统会自动将请求拆分为多个批次进行处理，避免单次请求过大导致的API配额错误。

### 配置方式

在 `config/settings.yaml` 中的 `graph_builder` 配置段：

```yaml
graph_builder:
  embedding_batch_size: 5000  # 每批次处理的实体嵌入数量和LLM仲裁数量，避免API配额错误
```

### 工作原理

1. **实体消歧阶段**：
   - 当图中节点数量超过 `embedding_batch_size` 时，自动拆分为多个批次
   - 每个批次生成独立的请求文件（如 `disambiguation_requests_batch_1.jsonl`）
   - 依次提交和处理每个批次，最后合并结果

2. **实体合并阶段**：
   - **嵌入向量生成**：当实体数量超过 `embedding_batch_size` 时，分批生成嵌入向量
   - **LLM仲裁请求**：当候选簇数量超过 `embedding_batch_size` 时，分批生成合并请求
   - 每个批次独立处理，最后合并所有结果

### 日志示例

```
⚙️ 实体数量 12000 超过批次大小 5000，将拆分为 3 个批次处理
🔄 处理批次 1/3：实体 1-5000
✅ 批次 1/3 完成，获得 5000 个嵌入向量
🔄 处理批次 2/3：实体 5001-10000
✅ 批次 2/3 完成，获得 5000 个嵌入向量
🔄 处理批次 3/3：实体 10001-12000
✅ 批次 3/3 完成，获得 2000 个嵌入向量
🎉 所有批次处理完成，共获得 12000 个嵌入向量
```

## 功能2：人工审核功能

### 功能说明

在实体合并阶段，系统可以随机抽样候选簇进行人工审核，并对比大模型的合并决策与人工决策，生成对比报告和可视化图表。

### 配置方式

在 `config/settings.yaml` 中的 `graph_builder` 配置段添加以下配置：

```yaml
graph_builder:
  # 人工审核配置
  enable_manual_review: False  # 是否启用人工审核功能（默认关闭）
  manual_review_sample_size: 5  # 人工审核抽样数量
  manual_review_output_dir: "data/reports/manual_review"  # 人工审核输出目录
```

### 使用方法

#### 启用人工审核

将 `enable_manual_review` 设置为 `True`：

```yaml
graph_builder:
  enable_manual_review: True
  manual_review_sample_size: 5
```

#### 审核流程

1. **运行图谱构建流程**：
   ```bash
   python app/run_indexing.py
   ```

2. **进入人工审核界面**：
   - 系统会随机抽取指定数量的候选簇
   - 对于每个候选簇，系统会显示实体详细信息
   
3. **审核提示示例**：
   ```
   ================================================================================
   候选簇 #42
   ================================================================================
   
   实体 1:
     ID: paper_123-e-45
     名称: γ' precipitate
     类型: MATERIAL_PHASE
     描述: Ordered L12 structure precipitate in nickel-based superalloys...
     别名: gamma prime, γ', gamma-prime
   
   实体 2:
     ID: paper_234-e-67
     名称: gamma prime phase
     类型: MATERIAL_PHASE
     描述: The strengthening phase in superalloys with L12 crystal structure...
     别名: γ' phase, Ni3Al phase
   
   --------------------------------------------------------------------------------
   请判断这些实体是否应该合并:
   1. 合并 (这些实体指向同一概念)
   2. 保持分离 (这些实体是不同的概念)
   
   请输入选择 (1/2): 
   ```

4. **输入审核决策**：
   - 输入 `1` 选择合并，或输入 `2` 选择保持分离
   - 如果选择合并，还需提供规范名称和合并理由
   - 如果选择保持分离，需提供理由

5. **完成审核**：
   - 系统会对比LLM的决策与人工决策
   - 生成对比报告和可视化图表

### 输出结果

审核完成后，系统会在指定的输出目录生成以下文件：

1. **对比报告**：`entity_merge_review_report.json`
   ```json
   {
     "summary": {
       "total_reviewed": 5,
       "agreements": 4,
       "disagreements": 1,
       "agreement_rate": 0.8,
       "human_merge_count": 3,
       "llm_merge_count": 4,
       "human_merge_rate": 0.6,
       "llm_merge_rate": 0.8
     },
     "details": [...]
   }
   ```

2. **可视化对比图**：`entity_merge_review_comparison.png`
   - 包含3个子图：
     - 一致性对比饼图
     - 决策分布对比柱状图
     - 合并率对比柱状图

### 报告解读

#### 摘要统计

- `total_reviewed`: 总审核样本数
- `agreements`: 人工与LLM决策一致的数量
- `disagreements`: 人工与LLM决策不一致的数量
- `agreement_rate`: 一致率（百分比）
- `human_merge_count`: 人工决策合并的数量
- `llm_merge_count`: LLM决策合并的数量

#### 详细对比

报告中的 `details` 字段包含每个审核案例的详细信息：

```json
{
  "cluster_id": 42,
  "member_count": 2,
  "member_names": ["γ' precipitate", "gamma prime phase"],
  "human_decision": "merge",
  "human_canonical_name": "γ' phase",
  "human_rationale": "Same strengthening phase, just different naming conventions",
  "llm_decision": "merge",
  "llm_canonical_name": "γ' precipitate",
  "llm_rationale": "Both refer to the same ordered intermetallic phase",
  "agreement": true
}
```

### 控制台输出示例

```
🔍 启动人工审核流程，抽样数量: 5
已从 156 个候选簇中随机抽样 5 个
进度: 1/5
[显示候选簇信息并收集人工决策]
...
进度: 5/5
[显示候选簇信息并收集人工决策]

================================================================================
实体合并审核对比摘要
================================================================================

总审核样本数: 5
一致数量: 4
不一致数量: 1
一致率: 80.00%

人工决策:
  合并: 3 (60.00%)
  保持分离: 2

LLM决策:
  合并: 4 (80.00%)
  保持分离: 1

不一致案例详情 (1 个):
--------------------------------------------------------------------------------

案例 1 (簇 #78):
  成员: creep resistance, creep behavior
  人工决策: keep_separate
    理由: Different aspects - resistance is a property, behavior is a process
  LLM决策: merge
    规范名称: creep resistance
    理由: Both describe the same phenomenon of material deformation under stress

================================================================================

✅ 人工审核完成
```

## 注意事项

### 功能1注意事项

1. **批次大小设置**：
   - 建议根据API配额限制设置合适的 `embedding_batch_size`
   - 过小会导致批次过多，处理时间较长
   - 过大可能导致API配额错误

2. **文件管理**：
   - 分批处理会生成多个临时文件（`*_batch_1.jsonl`, `*_batch_2.jsonl` 等）
   - 这些文件会保留在 `data/cache` 目录中，可在处理完成后手动清理

### 功能2注意事项

1. **交互式操作**：
   - 人工审核需要在控制台进行交互式操作
   - 不适合在后台或自动化流程中使用
   - 建议在开发和调试阶段使用

2. **抽样数量**：
   - 建议设置为 5-10 个样本
   - 过多会导致审核时间过长
   - 过少可能无法准确评估LLM效果

3. **默认配置**：
   - 默认情况下人工审核功能是关闭的（`enable_manual_review: False`）
   - 只在需要评估LLM合并效果时才开启

4. **中文字体**：
   - 可视化图表会尝试使用中文字体（SimHei, Microsoft YaHei, SimSun）
   - 如果系统没有这些字体，中文可能显示为方块

## 代码集成示例

### 单独使用人工审核模块

如果需要在其他场景中使用人工审核功能，可以直接调用：

```python
from utils.entity_merge_review import run_entity_merge_review
from pathlib import Path

# 准备数据
graph = ...  # NetworkX图对象
clusters = [...]  # 候选簇列表
llm_groups = [...]  # LLM返回的合并分组

# 运行人工审核
report = run_entity_merge_review(
    graph=graph,
    clusters=clusters,
    llm_groups=llm_groups,
    sample_size=5,
    output_dir=Path("data/reports/manual_review")
)

# 查看报告
print(f"一致率: {report['summary']['agreement_rate']:.2%}")
```

## 常见问题

### Q1: 如何只使用批量控制功能，不使用人工审核？

**A**: 这是默认配置，无需修改。只要 `enable_manual_review: False`，就不会启动人工审核。

### Q2: 批量处理会影响最终结果吗？

**A**: 不会。批量处理只是将大任务拆分为小任务依次处理，最后合并所有结果，与一次性处理的结果完全相同。

### Q3: 人工审核的结果会影响最终的实体合并吗？

**A**: 当前版本中，人工审核仅用于评估LLM效果，不会直接影响实际的合并结果。未来可以扩展为支持人工决策覆盖LLM决策。

### Q4: 如何清理分批生成的临时文件？

**A**: 可以手动删除 `data/cache` 目录中以 `_batch_` 命名的文件，或者在流程完成后运行清理脚本。

### Q5: 人工审核可以中断后继续吗？

**A**: 当前版本不支持断点续传。如果中途中断，需要重新开始审核流程。

## 更新日志

- **2025-01-09**: 初始版本发布
  - 实现批量请求数量控制功能
  - 实现人工审核与LLM对比功能
  - 支持可视化对比报告生成

