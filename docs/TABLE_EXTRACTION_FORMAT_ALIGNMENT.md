# Table Extraction 格式对齐修改总结

## 修改概述

已成功修改 `table_extraction.py` 和 `settings.yaml`，使表格提取的中间文件格式与 `loader.py` 生成的格式完全一致，并将所有路径配置写入 `settings.yaml`。

## 主要修改

### 1. Settings.yaml 配置

**新增 `table_extraction` 配置块**：

```yaml
table_extraction:
  input_dir: "data/processed_jsons"  # VLM 解析后的 JSON 文件目录
  output_dir: "data/chunks"  # 中间文件保存位置（与 loader 一致）
  output_filename: "table_units.jsonl"  # 表格 text units 文件
  graph_output_dir: "data/graphs/extracted"  # 图谱输出目录
  graph_output_filename: "extracted_table_graph.jsonl"  # 表格图谱文件
  requests_dir: "data/cache"  # 批量请求文件目录
  requests_filename: "table_extraction_requests.jsonl"  # 批量请求文件名
  prompt_path: "config/prompts/table_to_graph.md"
```

### 2. 中间文件格式对齐

#### Loader 的输出格式（text_units.jsonl）：
```json
{
  "id": "chunk-{hash}",
  "document_id": "doc-{hash}",
  "text": "chunk text content...",
  "metadata": {
    "chunk_index": 0,
    "source_filename": "xxx.json",
    "pages": [1, 2],
    "blocks": ["page_1_block_2", "page_1_block_3"]
  }
}
```

#### Table Extraction 的输出格式（table_units.jsonl）：
```json
{
  "id": "chunk-{hash}",
  "document_id": "doc-{hash}",
  "text": "Caption: xxx\nSummary: xxx\nData: [[...], [...]]",
  "metadata": {
    "source_filename": "xxx.json",
    "pages": [1],
    "blocks": ["page_1_block_3"]
  }
}
```

**关键对齐点**：
- ✅ 相同的顶层字段：`id`, `document_id`, `text`, `metadata`
- ✅ 相同的元数据结构：`source_filename`, `pages`, `blocks`
- ✅ 唯一区别：text 字段保留了表格 table 的描述性信息（caption、summary、data）

### 3. Text 字段内容

**Text 字段现在包含**：
1. **Caption**（如果有）：表格标题
2. **Summary**（如果有）：表格摘要
3. **Data**：表格数据的 JSON 表示

**示例**：
```
Caption: Material Properties Comparison
Summary: Yield strength and tensile strength across different alloys
Data: [["Material", "YS (MPa)", "TS (MPa)"], ["Alloy 718", "1000", "1200"], ["Alloy 625", "900", "1100"]]
```

### 4. 代码结构变更

#### TableProcessor 初始化
```python
# 之前：硬编码路径
self.input_dir = PROJECT_ROOT / self.config['loader']['source_json_dir']
self.output_dir = PROJECT_ROOT / "data/graphs/enriched"

# 现在：从 settings 读取
table_config = self.config.get("table_extraction", {})
self.input_dir = PROJECT_ROOT / table_config.get("input_dir", "data/processed_jsons")
self.tables_units_file = self.tables_units_dir / table_config.get("output_filename", "table_units.jsonl")
self.graph_output_file = self.graph_output_dir / table_config.get("graph_output_filename", "extracted_table_graph.jsonl")
```

#### 文件输出
- **中间文件**：`data/chunks/table_units.jsonl`（与 loader 输出在同一目录）
- **图谱文件**：`data/graphs/extracted/extracted_table_graph.jsonl`（与 extraction_qwen 输出在同一目录）

### 5. 数据流

```
原始 JSON (VLM 解析)
    ↓
[prepare_batch_requests]
    ↓
┌─────────────────────────────────┬────────────────────────────────┐
│ table_units.jsonl               │ table_extraction_requests.jsonl│
│ (中间文件，与 loader 格式一致)      │ (批量请求)                        │
└─────────────────────────────────┴────────────────────────────────┘
    ↓                                   ↓
    │                            [上传 + 批量推理]
    │                                   ↓
    │                            [下载结果]
    │                                   ↓
    └──────────────────>  [合并] ───────┘
                              ↓
                extracted_table_graph.jsonl
                (最终图谱输出)
```

## 与 Loader/Extraction 的兼容性

### Text Units 格式对比

| 字段 | Loader (text) | Table Extraction (table) | 兼容性 |
|------|---------------|-------------------------|--------|
| id | chunk-{hash} | chunk-{hash} | ✅ 相同格式 |
| document_id | doc-{hash} | doc-{hash} | ✅ 相同格式 |
| text | chunk text | Caption + Summary + Data | ✅ 都是字符串 |
| metadata.source_filename | xxx.json | xxx.json | ✅ 相同 |
| metadata.pages | [1, 2, ...] | [1] | ✅ 相同类型 |
| metadata.blocks | ["page_1_block_2", ...] | ["page_1_block_3"] | ✅ 相同类型 |
| metadata.chunk_index | 0, 1, 2, ... | (不包含) | ⚠️ text 特有 |

### Graph 输出格式对比

| 字段 | Extraction (text) | Table Extraction (table) | 兼容性 |
|------|------------------|-------------------------|--------|
| id | chunk-{hash} | chunk-{hash} | ✅ 相同 |
| graph | {...} | {...} | ✅ 相同 |

## 合并方式

### 1. 合并 Text Units（用于向量检索）
```bash
cat data/chunks/text_units.jsonl data/chunks/table_units.jsonl > data/chunks/all_units.jsonl
```

### 2. 合并 Graph 数据（用于图谱构建）
```bash
cat data/graphs/extracted/extracted_graph.jsonl data/graphs/extracted/extracted_table_graph.jsonl > data/graphs/extracted/all_graphs.jsonl
```

### 3. 后续流程可以直接使用合并后的文件
```python
# graph_builder_qwen.py 可以直接读取
with open("data/graphs/extracted/all_graphs.jsonl") as f:
    for line in f:
        data = json.loads(line)
        # 处理 id 和 graph 数据
```

## 配置使用

### 默认配置
所有路径都在 `settings.yaml` 中定义，无需修改代码即可更改路径。

### 自定义配置示例
```yaml
table_extraction:
  input_dir: "data/my_custom_jsons"
  output_dir: "data/my_chunks"
  output_filename: "my_table_units.jsonl"
  graph_output_dir: "data/my_graphs"
  graph_output_filename: "my_table_graphs.jsonl"
```

## 验证检查

### 1. 检查 table_units.jsonl 格式
```python
import json

with open("data/chunks/table_units.jsonl", 'r', encoding='utf-8') as f:
    for line in f:
        unit = json.loads(line)
        print(f"ID: {unit['id']}")
        print(f"Document ID: {unit['document_id']}")
        print(f"Text preview: {unit['text'][:100]}...")
        print(f"Metadata: {unit['metadata']}")
        break
```

### 2. 验证与 text_units 的兼容性
```python
# 两种 units 可以使用相同的读取逻辑
def load_units(file_path):
    units = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            units.append(json.loads(line))
    return units

text_units = load_units("data/chunks/text_units.jsonl")
table_units = load_units("data/chunks/table_units.jsonl")

# 验证字段一致性
assert set(text_units[0].keys()) == set(table_units[0].keys())
```

## 优势

1. ✅ **格式统一**：table_units 与 text_units 使用相同的数据结构
2. ✅ **易于合并**：可以直接合并为一个文件，后续处理无需区分来源
3. ✅ **配置集中**：所有路径在 settings.yaml 中统一管理
4. ✅ **信息完整**：text 字段保留了表格的 caption、summary 和 data
5. ✅ **向后兼容**：不影响现有的 extraction 和 graph_builder 流程

## 使用方法

```python
from core.pipeline_qwen.table_extraction import TableProcessor

# 使用默认配置
processor = TableProcessor()
processor.run()

# 输出文件：
# - data/chunks/table_units.jsonl (中间文件)
# - data/graphs/extracted/extracted_table_graph.jsonl (图谱输出)
```

## 注意事项

1. **Text 字段内容**：虽然格式与 loader 一致，但内容是表格的描述性文字，不是分块的纯文本
2. **chunk_index**：table_units 不包含 chunk_index（因为表格不分块）
3. **ID 唯一性**：确保 text 和 table 的 chunk_id 不会冲突（通过在 hash 输入中加入 "table" 标识）
4. **合并顺序**：合并时建议先 text 后 table，便于追踪

## 后续计划

基于相同的模式，可以扩展到 image extraction：

```yaml
image_extraction:
  input_dir: "data/processed_jsons"
  output_dir: "data/chunks"
  output_filename: "image_units.jsonl"
  graph_output_dir: "data/graphs/extracted"
  graph_output_filename: "extracted_image_graph.jsonl"
  requests_dir: "data/cache"
  requests_filename: "image_extraction_requests.jsonl"
  prompt_paths:
    chart: "config/prompts/chart_to_graph.md"
    schematic: "config/prompts/schematic_to_graph.md"
    photograph: "config/prompts/photograph_to_graph.md"
```

