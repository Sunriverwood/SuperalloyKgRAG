# 索引流水线执行指南

> 更新时间：2026-03-24（与当前主流程入口对齐）
> 
> 说明：本轮文档整理不包含 `draw/` 与 `visualizations/` 目录。

## 概述

`run_index_qwen.py` 是 SuperalloyKgRAG 当前主索引流水线执行脚本，实现从 PDF 文档到知识图谱向量存储的端到端数据处理流程。

## 流水线步骤

该脚本按顺序执行以下 4 个步骤：

### 步骤 1: OCR 解析 (vlm_pdf_parser)
- **输入**: `data/original_data/books/*.pdf`（默认，可在 `config/settings.yaml` 中修改）
- **输出**: `data/processed_jsons/*.json`
- **功能**: 使用 Vision-Language Model 将 PDF 转换为结构化 JSON
- **当前限制**: 该步骤当前仅支持 Gemini API；Qwen API 不支持直接输入 PDF 进行 OCR

### 步骤 2: 富化提取 (enrich_extraction)
- **输入**: `data/processed_jsons/*.json`（及其他可选抽取来源）
- **输出**: `data/graphs/enriched/enriched_graph.jsonl`
- **功能**: 融合文本/摘要/图片/表格抽取结果，生成富化图谱

### 步骤 3: 图谱构建 (graph_builder)
- **输入**: `data/graphs/enriched/enriched_graph.jsonl`
- **输出**: 
  - `data/graphs/final_graph.json`
  - `data/reports/community_summaries.jsonl`
- **功能**: 实体消歧、实体合并、社区发现、社区摘要

### 步骤 4: 向量化存储 (embedding)
- **输入**: 
  - `data/graphs/final_graph.json`
  - `data/reports/community_summaries.jsonl`
- **输出**: 
  - `data/embeddings/enriched.db`（默认 LanceDB 向量数据库，受 `embedding.output_db_path` 控制）
- **功能**: 将图谱数据嵌入向量数据库，支持相似度搜索

## 使用方法

### 1. 完整流水线执行

执行全部 4 个步骤：

```bash
python app/run_index_qwen.py
```

### 2. 从指定步骤开始执行

从步骤 3 开始执行到步骤 4：

```bash
python app/run_index_qwen.py --start 3
```

### 3. 执行指定范围的步骤

仅执行步骤 2-3：

```bash
python app/run_index_qwen.py --start 2 --end 3
```

### 4. 执行单个步骤

仅执行步骤 3（图谱构建）：

```bash
python app/run_index_qwen.py --step 3
```

### 5. 重置状态后执行

重置流水线状态，从头开始执行：

```bash
python app/run_index_qwen.py --reset
```

### 6. 断点续传

从上次中断的地方继续执行：

```bash
python app/run_index_qwen.py --resume
```

## 核心功能

### 1. 步骤依赖验证

每个步骤执行前会自动验证输入文件是否存在：

- **步骤 1**: 检查 `vlm_parser.input_dir`（默认 `data/original_data/books/`）中是否有 PDF 文件
- **步骤 2**: 检查 `data/processed_jsons/` 中是否有 JSON 文件
- **步骤 3**: 检查 `data/graphs/enriched/enriched_graph.jsonl` 是否存在且非空
- **步骤 4**: 检查 `data/graphs/final_graph.json` 和 `data/reports/community_summaries.jsonl` 是否存在

如果依赖验证失败，脚本会提示错误信息并终止执行。

### 2. 错误恢复策略

脚本会在 `data/cache/pipeline_state.json` 中记录每个步骤的执行状态：

```json
{
  "last_completed_step": 3,
  "last_run_time": "2025-11-19T14:30:00",
  "steps": {
    "ocr_parsing": {
      "completed": true,
      "timestamp": "2025-11-19T12:00:00",
      "duration": "3600.00s"
    },
    "enrich_extraction": {
      "completed": true,
      "timestamp": "2025-11-19T13:00:00",
      "duration": "120.50s"
    },
    "graph_building": {
      "completed": true,
      "timestamp": "2025-11-19T14:30:00",
      "duration": "5400.00s"
    },
    "vector_embedding": {
      "completed": false,
      "timestamp": null,
      "duration": null
    }
  }
}
```

**优势**：
- 避免重复执行已完成的耗时步骤（特别是 OCR 和 LLM 批处理）
- 支持从失败点重启
- 记录每步执行时间，便于性能分析

### 3. 统一日志管理

所有日志会输出到：
- **控制台**: 实时查看执行进度
- **日志文件**: `logs/superalloyKgRAG.log`

日志格式：
```
2025-11-19 14:30:00 [INFO] IndexingPipeline - ✅ 步骤 3 (graph_building) 已完成，耗时: 5400.00s
```

## 命令行参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `--start` | int (1-4) | 起始步骤，默认为 1 |
| `--end` | int (1-4) | 结束步骤，默认为 4 |
| `--step` | int (1-4) | 仅执行指定的单个步骤 |
| `--reset` | flag | 重置流水线状态后执行 |
| `--resume` | flag | 从上次中断处继续执行 |
| `--ablation` | str | 指定消融配置名称（可选） |

## 常见使用场景

### 场景 1: 首次处理新数据

```bash
# 1. 将 PDF 文件放入 data/original_data/books/
# 2. 执行完整流水线
python app/run_index_qwen.py
```

### 场景 2: 更新已有数据

```bash
# 1. 添加新的 PDF 文件到 data/original_data/books/
# 2. 仅执行 OCR 解析
python app/run_index_qwen.py --step 1
# 3. 继续执行后续步骤
python app/run_index_qwen.py --start 2
```

### 场景 3: 流水线中断后恢复

```bash
# 假设在步骤 3 时因网络问题中断
# 直接使用 --resume 继续
python app/run_index_qwen.py --resume
```

### 场景 4: 调试特定步骤

```bash
# 仅重新执行图谱构建步骤
python app/run_index_qwen.py --step 3
```

### 场景 5: 重新处理全部数据

```bash
# 清空状态，重新执行所有步骤
python app/run_index_qwen.py --reset
```

## 配置说明

流水线读取 `config/settings.yaml` 中的配置，主要包括：

```yaml
# OCR 解析配置
vlm_parser:
  input_dir: "data/original_data/books/"
  output_dir: "data/processed_jsons/"
  batch_size: 20
  sleep_interval: 600

# 富化提取配置
enrich_extraction:
  extracted_dir: "data/graphs/extracted"
  enriched_dir: "data/graphs/enriched"
  enriched_filename: "enriched_graph.jsonl"

# 图谱构建配置
graph_builder:
  input_path: "data/graphs/enriched/enriched_graph.jsonl"
  enable_entity_merge: True
  entity_merge_topk: 10
  entity_merge_min_sim: 0.9
  community_importance_weight_alpha: 0.8

# 向量化配置
embedding:
  model: "text-embedding-v4"
  dimensionality: 1024

# 日志配置
logging:
  level: INFO
  log_file: "logs/superalloyKgRAG.log"
```

## 性能优化建议

1. **OCR 解析 (步骤 1)**:
   - 最耗时（数小时）
   - 使用批处理模式减少 API 调用
   - 建议在网络稳定时执行

2. **富化提取 (步骤 2)**:
   - 耗时较长（数小时）
   - 依赖 LLM 批处理
   - 可通过调整 `batch_size` 优化

3. **图谱构建 (步骤 3)**:
   - 实体合并较耗时
   - 可通过调整 `entity_merge_topk` 减少计算量

4. **向量化存储 (步骤 4)**:
   - 使用并行处理（3 个线程）
   - 性能受网络和 API 限速影响

## 故障排除

### 问题 1: 依赖验证失败

**错误信息**:
```
❌ 步骤3依赖验证失败: 富化图谱文件不存在
💡 提示: 请先运行步骤2 (富化提取)
```

**解决方案**: 按顺序执行前置步骤

### 问题 2: 批处理作业超时

**错误信息**:
```
❌ 步骤1执行失败: Batch job timeout
```

**解决方案**: 
- 检查网络连接
- 增加 `batch_polling_timeout_seconds` 配置
- 使用 `--resume` 继续执行

### 问题 3: API 配额耗尽

**错误信息**:
```
❌ API quota exceeded
```

**解决方案**:
- 等待配额重置
- 使用 `--resume` 从中断处继续

### 问题 4: 内存不足

**错误信息**:
```
MemoryError: Unable to allocate array
```

**解决方案**:
- 减小 `batch_size` 配置
- 分批处理 PDF 文件

## 最佳实践

1. **首次执行**: 使用小批量数据测试完整流水线
2. **生产环境**: 定期备份 `data/` 目录
3. **状态管理**: 不要手动修改 `pipeline_state.json`
4. **日志监控**: 定期检查 `logs/superalloyKgRAG.log`
5. **网络稳定性**: OCR 和富化提取步骤需要稳定网络

## 相关文件

- **主脚本**: `app/run_index_qwen.py`
- **配置文件**: `config/settings.yaml`
- **状态文件**: `data/cache/pipeline_state.json`
- **日志文件**: `logs/superalloyKgRAG.log`
- **各步骤模块**:
  - `core/vlm_pdf_parser.py`
  - `core/pipeline_qwen/enrich_extraction.py`
  - `core/pipeline_qwen/graph_builder_qwen.py`
  - `core/pipeline_qwen/embedding_qwen.py`

