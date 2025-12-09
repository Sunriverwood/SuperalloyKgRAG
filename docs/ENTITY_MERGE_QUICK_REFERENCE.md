# 实体合并功能快速参考

## 快速开始

### 功能1: 批量请求控制（已默认启用）

无需配置，已自动启用。如需调整批次大小：

```yaml
# config/settings.yaml
graph_builder:
  embedding_batch_size: 5000  # 调整为合适的值
```

### 功能2: 启用人工审核

```yaml
# config/settings.yaml
graph_builder:
  enable_manual_review: True
  manual_review_sample_size: 5
```

然后运行：

```bash
python app/run_indexing.py
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `embedding_batch_size` | 5000 | 批次大小，控制单次请求数量 |
| `enable_manual_review` | False | 是否启用人工审核 |
| `manual_review_sample_size` | 5 | 人工审核抽样数量 |
| `manual_review_output_dir` | `data/reports/manual_review` | 输出目录 |

## 输出文件

### 批量控制生成的文件

- `data/cache/disambiguation_requests_batch_1.jsonl`
- `data/cache/disambiguation_requests_batch_2.jsonl`
- `data/cache/entities_merge_requests_batch_1.jsonl`
- `data/cache/entities_merge_requests_batch_2.jsonl`
- ...

### 人工审核生成的文件

- `data/reports/manual_review/entity_merge_review_report.json` - JSON格式报告
- `data/reports/manual_review/entity_merge_review_comparison.png` - 可视化对比图

## 人工审核交互流程

1. 系统展示候选簇信息
2. 输入 `1` 选择合并，或 `2` 选择保持分离
3. 如果选择合并：
   - 输入规范名称
   - 输入合并理由
4. 如果选择保持分离：
   - 输入保持分离的理由
5. 重复直到完成所有抽样簇

## 常用命令

```bash
# 正常运行（批量控制自动启用）
python app/run_indexing.py

# 测试人工审核功能
python tests/test_entity_merge_review.py

# 检查依赖
python -c "import matplotlib; import networkx; print('OK')"
```

## 关键代码位置

| 功能 | 文件 | 行号范围 |
|------|------|----------|
| 批量请求创建 | `core/pipeline_qwen/graph_builder_qwen.py` | 206-300 |
| 实体合并批量控制 | `core/pipeline_qwen/graph_builder_qwen.py` | 587-650 |
| 消歧批量控制 | `core/pipeline_qwen/graph_builder_qwen.py` | 1045-1100 |
| 人工审核集成 | `core/pipeline_qwen/graph_builder_qwen.py` | 1195-1213 |
| 人工审核模块 | `utils/entity_merge_review.py` | 全文 |

## 常见问题

**Q: 批量控制会影响结果吗？**
A: 不会。只是将大任务拆分处理，结果完全相同。

**Q: 人工审核结果会应用到合并吗？**
A: 当前版本仅用于评估，不影响实际合并。

**Q: 如何清理临时文件？**
A: 手动删除 `data/cache/*_batch_*.jsonl` 文件。

**Q: 可视化图表中文乱码怎么办？**
A: 安装中文字体（SimHei、Microsoft YaHei、SimSun）。

## 更多信息

- 详细指南: `docs/ENTITY_MERGE_REVIEW_GUIDE.md`
- 实现总结: `docs/ENTITY_MERGE_IMPLEMENTATION_SUMMARY.md`
- 测试脚本: `tests/test_entity_merge_review.py`

