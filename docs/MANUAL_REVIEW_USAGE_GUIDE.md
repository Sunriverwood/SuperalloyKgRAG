# 实体合并人工审核使用指南

## 快速启用

### 步骤1: 修改配置文件

编辑 `config/settings.yaml`：

```yaml
graph_builder:
  # 人工审核配置
  enable_manual_review: True  # ← 改为 True
  manual_review_sample_size: 5  # 可调整抽样数量
  manual_review_output_dir: "data/reports/manual_review"
```

### 步骤2: 运行图谱构建

```bash
python app/run_indexing.py
# 或
python app/run_index_qwen.py
```

### 步骤3: 等待人工审核阶段

当看到以下日志时，表示进入人工审核阶段：

```
🔍 启动人工审核流程，抽样数量: 5
已从 156 个候选簇中随机抽样 5 个
```

## 交互式审核流程

### 典型交互示例

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

请输入选择 (1/2): 1
请输入合并后的规范名称: γ' phase
请简要说明合并理由: Same strengthening phase, different naming conventions
```

### 审核提示

**判断是否合并的依据：**

1. **应该合并**的情况：
   - 实体描述的是同一个概念
   - 只是命名方式不同（如缩写、全称、不同语言）
   - 同一材料/现象/方法的不同表述

2. **应该保持分离**的情况：
   - 虽然相关但是不同的概念
   - 一个是属性，一个是过程
   - 一个是材料，一个是性能
   - 上下位关系（如"金属"vs"钢铁"）

### 审核建议

- **专注于语义**：不要被表面的相似性迷惑
- **参考描述**：重点阅读description字段
- **考虑类型**：type字段提供重要线索
- **查看别名**：别名可以帮助判断是否同一概念
- **简洁理由**：理由简明扼要即可，主要用于记录

## 输出结果

### 自动生成的文件

审核完成后，在 `data/reports/manual_review/` 目录下生成：

1. **entity_merge_review_report.json** - 详细对比报告
   ```json
   {
     "summary": {
       "total_reviewed": 5,
       "agreements": 4,
       "disagreements": 1,
       "agreement_rate": 0.8,
       "human_merge_count": 3,
       "llm_merge_count": 4
     },
     "details": [...]
   }
   ```

2. **entity_merge_review_comparison.png** - 可视化对比图
   - 一致性饼图
   - 决策分布柱状图
   - 合并率对比图

### 控制台输出

```
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
    理由: Different concepts - resistance is a property, behavior is a process
  LLM决策: merge
    规范名称: creep resistance
    理由: Both describe the same phenomenon of material deformation under stress

================================================================================

✅ 人工审核完成
对比报告已保存到: data/reports/manual_review/entity_merge_review_report.json
可视化结果已保存到: data/reports/manual_review/entity_merge_review_comparison.png
```

## 运行时机

人工审核在以下时机自动触发：

1. ✅ 实体消歧完成
2. ✅ 嵌入向量生成完成
3. ✅ 候选簇构建完成
4. ✅ **LLM仲裁完成** ← 此时触发人工审核
5. ⏸️ 人工审核进行中...
6. ✅ 人工审核完成，生成报告
7. ✅ 应用LLM的合并决策（人工审核不影响此步骤）

**注意**：人工审核仅用于评估LLM效果，**不会改变实际的合并结果**。

## 配置参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_manual_review` | False | 是否启用人工审核 |
| `manual_review_sample_size` | 5 | 随机抽样数量，建议3-10 |
| `manual_review_output_dir` | `data/reports/manual_review` | 报告输出目录 |

## 注意事项

### 时间预估

- 每个候选簇审核：约1-2分钟
- 抽样5个：总计约5-10分钟
- 建议不要设置过大的sample_size

### 交互环境

- ✅ 必须在**交互式终端**中运行
- ❌ 不支持后台运行
- ❌ 不支持自动化脚本

### 数据要求

- 必须有候选簇（clusters不为空）
- 必须有LLM仲裁结果（groups不为空）
- 如果没有候选簇，会自动跳过人工审核

### 中断恢复

- 当前版本**不支持断点续传**
- 如果中途退出，需要重新开始
- 建议一次性完成所有审核

## 实际使用场景

### 场景1: 评估LLM合并质量

**目的**：了解LLM在实体合并任务上的准确率

**操作**：
1. 启用人工审核
2. 认真审核每个候选簇
3. 查看一致率（agreement_rate）

**评估标准**：
- 一致率 > 90%：LLM表现优秀
- 一致率 70-90%：LLM表现良好
- 一致率 < 70%：需要调整参数或提示词

### 场景2: 调试合并参数

**目的**：找到最优的相似度阈值

**操作**：
1. 设置不同的 `entity_merge_min_sim`
2. 每次都进行人工审核
3. 对比不同阈值下的一致率

**建议阈值**：
- 保守：0.92-0.95（较少合并，准确率高）
- 平衡：0.88-0.92（推荐）
- 激进：0.82-0.88（较多合并，可能误合）

### 场景3: 优化提示词

**目的**：改进entity_disambiguation.md提示词

**操作**：
1. 记录不一致案例
2. 分析LLM的错误模式
3. 调整提示词中的指导语
4. 重新运行并对比效果

## 高级用法

### 调整抽样数量

```yaml
# 快速评估（3个样本）
manual_review_sample_size: 3

# 标准评估（5个样本，推荐）
manual_review_sample_size: 5

# 深度评估（10个样本）
manual_review_sample_size: 10
```

### 自定义输出目录

```yaml
# 按日期组织
manual_review_output_dir: "data/reports/manual_review/2025-01-09"

# 按实验名称组织
manual_review_output_dir: "data/reports/manual_review/exp_threshold_0.9"
```

### 查看历史对比

```bash
# 列出所有历史报告
ls data/reports/manual_review/

# 查看特定报告
cat data/reports/manual_review/entity_merge_review_report.json | jq '.summary'

# 对比多次实验
python -c "
import json
from pathlib import Path

for report in Path('data/reports/manual_review').glob('*/entity_merge_review_report.json'):
    with open(report) as f:
        data = json.load(f)
        print(f'{report.parent.name}: {data[\"summary\"][\"agreement_rate\"]:.2%}')
"
```

## 常见问题

### Q: 人工审核会影响最终的合并结果吗？

**A**: 不会。人工审核仅用于评估和对比，实际的合并仍然基于LLM的决策。

### Q: 可以只进行人工审核，不应用LLM的合并吗？

**A**: 当前版本不支持。未来可以扩展为：
- 导出人工审核结果
- 手动应用人工决策
- 或混合使用LLM和人工决策

### Q: 抽样是完全随机的吗？

**A**: 是的。使用 `random.sample()` 进行均匀随机抽样，确保每次运行可能抽到不同的簇。

### Q: 如何让抽样结果可复现？

**A**: 可以在代码中设置随机种子（需要修改代码）：
```python
# 在 EntityMergeReviewer.__init__ 中添加
random.seed(42)  # 固定种子
```

### Q: 可以审核所有候选簇吗？

**A**: 可以，将 `manual_review_sample_size` 设置为一个很大的数（如10000），系统会自动限制为实际的候选簇数量。但不建议这样做，因为会非常耗时。

## 最佳实践

1. **首次使用**：设置sample_size=3，快速体验流程
2. **正式评估**：设置sample_size=5，获得较准确的评估
3. **定期评估**：每次调整参数后都进行一次人工审核
4. **记录对比**：保存不同配置下的审核报告，便于对比
5. **关注不一致案例**：重点分析LLM的错误模式

## 总结

人工审核功能让您能够：
- ✅ 评估LLM的实体合并质量
- ✅ 发现LLM的错误模式
- ✅ 优化合并参数和提示词
- ✅ 生成详细的对比报告
- ✅ 可视化展示评估结果

**启用方式**：只需将 `enable_manual_review` 改为 `True`，然后正常运行 `run_indexing.py` 即可！

---

**文档版本**: v1.0  
**最后更新**: 2025-01-09

