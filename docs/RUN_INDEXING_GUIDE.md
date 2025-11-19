# 索引流水线执行指南

## 概述

`run_indexing.py` 是 SuperalloyKgRAG 项目的核心索引流水线执行脚本，实现从 PDF 文档到知识图谱向量存储的端到端数据处理流程。

## 流水线步骤

该脚本按顺序执行以下 5 个步骤：

### 步骤 1: OCR 解析 (vlm_pdf_parser)
- **输入**: `data/raw_pdfs/*.pdf`
- **输出**: `data/processed_jsons/*.json`
- **功能**: 使用 Vision-Language Model 将 PDF 转换为结构化 JSON

### 步骤 2: 文本分块 (loader)
- **输入**: `data/processed_jsons/*.json`
- **输出**: `data/chunks/text_units.jsonl`
- **功能**: 将 JSON 文档切分为可处理的文本单元

### 步骤 3: 三元组提取 (extraction)
- **输入**: `data/chunks/text_units.jsonl`
- **输出**: `data/graphs/extracted/extracted_graph.jsonl`
- **功能**: 从文本块中提取实体和关系（三元组）

### 步骤 4: 图谱构建 (graph_builder)
- **输入**: `data/graphs/extracted/extracted_graph.jsonl`
- **输出**: 
  - `data/graphs/final_graph.json`
  - `data/reports/community_summaries.jsonl`
- **功能**: 实体消歧、实体合并、社区发现、社区摘要

### 步骤 5: 向量化存储 (embedding)
- **输入**: 
  - `data/graphs/final_graph.json`
  - `data/reports/community_summaries.jsonl`
- **输出**: 
  - `data/embeddings/embeddings.db/` (LanceDB 向量数据库)
- **功能**: 将图谱数据嵌入向量数据库，支持相似度搜索

## 使用方法

### 1. 完整流水线执行

执行全部 5 个步骤：

```bash
python app/run_indexing.py
```

### 2. 从指定步骤开始执行

从步骤 3 开始执行到步骤 5：

```bash
python app/run_indexing.py --start 3
```

### 3. 执行指定范围的步骤

仅执行步骤 2-4：

```bash
python app/run_indexing.py --start 2 --end 4
```

### 4. 执行单个步骤

仅执行步骤 3（三元组提取）：

```bash
python app/run_indexing.py --step 3
```

### 5. 重置状态后执行

重置流水线状态，从头开始执行：

```bash
python app/run_indexing.py --reset
```

### 6. 断点续传

从上次中断的地方继续执行：

```bash
python app/run_indexing.py --resume
```

## 核心功能

### 1. 步骤依赖验证

每个步骤执行前会自动验证输入文件是否存在：

- **步骤 1**: 检查 `data/raw_pdfs/` 中是否有 PDF 文件
- **步骤 2**: 检查 `data/processed_jsons/` 中是否有 JSON 文件
- **步骤 3**: 检查 `data/chunks/text_units.jsonl` 是否存在且非空
- **步骤 4**: 检查 `data/graphs/extracted/extracted_graph.jsonl` 是否存在且非空
- **步骤 5**: 检查 `data/graphs/final_graph.json` 和 `data/reports/community_summaries.jsonl` 是否存在

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
    "text_chunking": {
      "completed": true,
      "timestamp": "2025-11-19T13:00:00",
      "duration": "120.50s"
    },
    "triple_extraction": {
      "completed": true,
      "timestamp": "2025-11-19T14:30:00",
      "duration": "5400.00s"
    },
    "graph_building": {
      "completed": false,
      "timestamp": null,
      "duration": null
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
2025-11-19 14:30:00 [INFO] IndexingPipeline - ✅ 步骤 3 (triple_extraction) 已完成，耗时: 5400.00s
```

## 命令行参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `--start` | int (1-5) | 起始步骤，默认为 1 |
| `--end` | int (1-5) | 结束步骤，默认为 5 |
| `--step` | int (1-5) | 仅执行指定的单个步骤 |
| `--reset` | flag | 重置流水线状态后执行 |
| `--resume` | flag | 从上次中断处继续执行 |

## 常见使用场景

### 场景 1: 首次处理新数据

```bash
# 1. 将 PDF 文件放入 data/raw_pdfs/
# 2. 执行完整流水线
python app/run_indexing.py
```

### 场景 2: 更新已有数据

```bash
# 1. 添加新的 PDF 文件到 data/raw_pdfs/
# 2. 仅执行 OCR 解析
python app/run_indexing.py --step 1
# 3. 继续执行后续步骤
python app/run_indexing.py --start 2
```

### 场景 3: 流水线中断后恢复

```bash
# 假设在步骤 3 时因网络问题中断
# 直接使用 --resume 继续
python app/run_indexing.py --resume
```

### 场景 4: 调试特定步骤

```bash
# 仅重新执行图谱构建步骤
python app/run_indexing.py --step 4
```

### 场景 5: 重新处理全部数据

```bash
# 清空状态，重新执行所有步骤
python app/run_indexing.py --reset
```

## 配置说明

流水线读取 `config/settings.yaml` 中的配置，主要包括：

```yaml
# OCR 解析配置
vlm_parser:
  input_dir: "data/raw_pdfs/"
  output_dir: "data/processed_jsons/"
  batch_size: 20
  sleep_interval: 600

# 文本分块配置
loader:
  chunk_size: 500
  chunk_overlap: 100
  output_format: "jsonl"

# 三元组提取配置
extraction:
  prompt_path: "config/prompts/text_to_graph.md"
  output_filename: "extracted_graph.jsonl"

# 图谱构建配置
graph_builder:
  enable_entity_merge: True
  entity_merge_topk: 10
  entity_merge_min_sim: 0.82
  community_importance_weight_alpha: 0.6

# 向量化配置
embedding:
  model: "gemini-embedding-001"
  dimensionality: 768

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

2. **三元组提取 (步骤 3)**:
   - 耗时较长（数小时）
   - 依赖 LLM 批处理
   - 可通过调整 `batch_size` 优化

3. **图谱构建 (步骤 4)**:
   - 实体合并较耗时
   - 可通过调整 `entity_merge_topk` 减少计算量

4. **向量化存储 (步骤 5)**:
   - 使用并行处理（3 个线程）
   - 性能受网络和 API 限速影响

## 故障排除

### 问题 1: 依赖验证失败

**错误信息**:
```
❌ 步骤3依赖验证失败: 文本单元文件不存在
💡 提示: 请先运行步骤2 (文本分块)
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
- 减小 `chunk_size` 配置
- 减小 `batch_size` 配置
- 分批处理 PDF 文件

## 最佳实践

1. **首次执行**: 使用小批量数据测试完整流水线
2. **生产环境**: 定期备份 `data/` 目录
3. **状态管理**: 不要手动修改 `pipeline_state.json`
4. **日志监控**: 定期检查 `logs/superalloyKgRAG.log`
5. **网络稳定性**: OCR 和提取步骤需要稳定网络

## 相关文件

- **主脚本**: `app/run_indexing.py`
- **配置文件**: `config/settings.yaml`
- **状态文件**: `data/cache/pipeline_state.json`
- **日志文件**: `logs/superalloyKgRAG.log`
- **各步骤模块**:
  - `core/vlm_pdf_parser.py`
  - `core/pipeline/loader.py`
  - `core/pipeline/extraction.py`
  - `core/pipeline/graph_builder.py`
  - `core/pipeline/embedding.py`

