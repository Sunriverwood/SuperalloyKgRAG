# Table Extraction 批量推理模式实现

## 修改总结

已成功将 `table_extraction.py` 从单次调用模式改为批量推理模式，完全仿照 `extraction_qwen.py` 的实现。

## 主要改动

### 1. 添加依赖
```python
import time  # 用于轮询作业状态
```

### 2. 新增类属性
```python
# 批量请求文件路径
self.requests_dir = PROJECT_ROOT / "data/cache/table_requests"
self.batch_request_path = self.requests_dir / "table_extraction_requests.jsonl"

# 中间数据文件：存储 table block 信息
self.tables_data_path = self.requests_dir / "tables_data.jsonl"
```

### 3. 删除的方法
- ❌ `_call_llm_for_table()` - 单次 LLM 调用
- ❌ `process_table_block()` - 单个表格处理

### 4. 新增的方法
- ✅ `prepare_batch_requests()` - 准备批量请求文件

### 5. 重写的方法
- ✅ `run()` - 完整的批量推理流程

## 批量推理流程

### 步骤 1: 准备批量请求
```python
def prepare_batch_requests(self) -> int:
    """
    遍历所有 JSON 文件，收集表格数据：
    1. 为每个表格生成 chunk_id
    2. 构建批量请求 (custom_id + messages)
    3. 保存表格元数据到中间文件
    """
```

**输出文件**：
- `table_extraction_requests.jsonl` - 批量请求文件
- `tables_data.jsonl` - 表格元数据（用于最后构建输出）

### 步骤 2: 上传批量请求文件
```python
uploaded_file = self.client.files.create(
    file=file_obj,
    purpose="batch"
)
```

### 步骤 3: 创建批量作业
```python
file_batch_job = self.client.batches.create(
    input_file_id=uploaded_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h"
)
```

### 步骤 4: 轮询作业状态
```python
while True:
    batch_job_status = self.client.batches.retrieve(batch_id=job_id)
    if batch_job_status.status in completed_states:
        break
    time.sleep(sleep_interval)
```

### 步骤 5: 下载并处理结果
```python
file_content = self.client.files.content(output_file_id).text

# 解析结果，匹配元数据，构建最终输出
for line in file_content.strip().split('\n'):
    result = json.loads(line)
    chunk_id = result.get("custom_id")
    # ... 提取 graph_data
    # ... 从 tables_metadata 获取元数据
    # ... 构建最终输出
```

## 数据流

```
输入 JSON 文件
    ↓
[prepare_batch_requests]
    ↓
table_extraction_requests.jsonl (批量请求)
tables_data.jsonl (元数据)
    ↓
[上传 + 创建作业]
    ↓
[轮询等待]
    ↓
[下载结果]
    ↓
[匹配元数据 + 构建输出]
    ↓
extracted_table.jsonl (最终输出)
```

## 输出格式

与 `extraction_qwen.py` 保持完全一致：

```json
{
  "id": "chunk-{hash}",
  "document_id": "doc-{hash}",
  "graph": {
    "entities": [...],
    "relationships": [...]
  },
  "metadata": {
    "chunk_type": "table",
    "document_id": "doc-{hash}",
    "source_filename": "xxx.json",
    "pages": [1],
    "blocks": ["page_1_block_3"],
    "caption": "...",
    "summary": "..."
  }
}
```

## 关键特性

### 1. 原始数据保留
- ✅ 直接将完整的 `block` 数据传给 LLM
- ✅ 不做任何人工转换或重组
- ✅ LLM 可以看到 type, block_id, caption, summary, data 等所有字段

### 2. 批量请求格式
```json
{
  "custom_id": "chunk-{hash}",
  "method": "POST",
  "url": "/v1/chat/completions",
  "body": {
    "model": "qwen-plus",
    "messages": [
      {"role": "system", "content": "{prompt_template}"},
      {"role": "user", "content": "{block_json}"}
    ],
    "temperature": 0.1,
    "top_p": 0.9,
    "response_format": {"type": "json_object"}
  }
}
```

### 3. 元数据中间文件
```json
{
  "id": "chunk-{hash}",
  "document_id": "doc-{hash}",
  "source_filename": "xxx.json",
  "page_number": 1,
  "block_id": "page_1_block_3",
  "caption": "...",
  "summary": "...",
  "block": {...}  // 完整的原始 block
}
```

## 优势

### 1. 性能提升
- **批量处理**：一次性提交所有表格，避免逐个调用
- **并行推理**：服务端并行处理多个请求
- **减少等待**：不需要为每个表格单独等待

### 2. 成本优化
- **批量定价**：通常比单次调用更便宜（具体看服务商）
- **减少请求数**：降低 API 调用开销

### 3. 可靠性
- **任务持久化**：作业创建后即使断网也能继续
- **可恢复**：可以通过 job_id 恢复查��状态
- **错误隔离**：单个表格失败不影响其他表格

### 4. 与现有流程一致
- ✅ 与 `extraction_qwen.py` 使用相同的模式
- ✅ 输出格式完全兼容
- ✅ 可以直接合并到后续的图谱构建流程

## 使用方法

```python
from core.pipeline_qwen.table_extraction import TableProcessor

# 初始化
processor = TableProcessor()

# 运行批量提取
processor.run()

# 输出文件：data/graphs/enriched/extracted_table.jsonl
```

## 日志输出示例

```
============================================================
开始表格批量提取流程
============================================================
开始准备批量请求...
找到 5 个 JSON 文件
✅ 成功创建批量请求文件，共 23 个表格: table_extraction_requests.jsonl
📤 正在上传批量请求文件: table_extraction_requests.jsonl...
✅ 文件上传成功: file-abc123
🚀 正在创建批量作业...
✅ 批量作业已创建: batch-xyz789
⏳ 开始轮询作业 'batch-xyz789' 状态，每 60 秒检查一次...
  - 当前状态: in_progress
  - 当前状态: in_progress
  - 当前状态: completed
✅ 作业成功完成！
📥 正在下载结果文件: file-def456
🎉 结果处理完成！成功处理 23 个表格，失败 0 个。
💾 最终图谱数据已保存至: data/graphs/enriched/extracted_table.jsonl
```

## 配置要求

确保 `config/settings.yaml` 包含以下配置：

```yaml
llm:
  model: "qwen-plus"
  temperature: 0.1
  top_p: 0.9

vlm_parser:
  sleep_interval: 60  # 轮询间隔（秒）

loader:
  source_json_dir: "data/processed_jsons"
```

## 环境变量

```bash
export QWEN_API_KEY="your-api-key"
```

## 注意事项

1. **首次运行**：会创建 `data/cache/table_requests/` 目录
2. **中间文件**：`tables_data.jsonl` 用于映射 chunk_id 到元数据，处理完成后可删除
3. **断点续传**：目前不支持，如果中断需要重新运行
4. **错误处理**：单个表格的错误不会导致整体失败，错误会被记录到日志

## 后续优化建议

1. **断点续传**：保存 job_id，支持从中断处恢复
2. **重试机制**：对失败的表格自动重试
3. **进度显示**：显示处理百分比
4. **结果缓存**：避免重复处理相同的表格

## 与 extraction_qwen 的对比

| 特性 | extraction_qwen | table_extraction |
|------|----------------|------------------|
| 输入 | text chunks (可能跨block) | table blocks (一对一) |
| Prompt | text_to_graph.md | table_to_graph.md |
| 批量模式 | ✅ | ✅ |
| 输出格式 | id + graph + metadata | ✅ 相同 |
| chunk_id | chunk-{hash} | ✅ 相同格式 |
| 元数据 | chunk_type="text" | chunk_type="table" |
| 可合并性 | ✅ | ✅ |

