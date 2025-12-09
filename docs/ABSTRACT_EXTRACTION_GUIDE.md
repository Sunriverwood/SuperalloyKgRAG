# Abstract Extraction Guide

## Overview

The `abstract_extraction.py` module extracts knowledge graph triples from research paper abstracts stored in Excel files. It follows the same pipeline pattern as `extraction_qwen.py` and `table_extraction.py`.

## Features

- **Excel Input**: Reads abstracts from Excel files (e.g., `data/papers/superalloy_research.xlsx`)
- **Flexible Column Mapping**: Automatically detects columns for Title, Abstract, Journal, Year, Author, and DOI
- **Intermediate File Generation**: Creates `abstract_units.jsonl` with proper metadata structure
- **Batch Processing**: Uses Alibaba Cloud Qwen API batch mode for efficient triple extraction
- **Graph Extraction**: Applies `text_to_graph.md` prompt to extract entities and relationships

## Workflow

```
Excel File (superalloy_research.xlsx)
    ↓
abstract_units.jsonl (chunked abstracts with metadata)
    ↓
extraction_abstract_requests.jsonl (batch API requests)
    ↓
Qwen Batch API Processing
    ↓
extracted_abstract_graph.jsonl (knowledge graph triples)
```

## File Structure

### Input Excel Format

The Excel file should contain columns with these names (case-insensitive):
- **Title**: Paper title (required for generating doc_id)
- **Abstract**: Paper abstract (required)
- **Journal**: Journal name (optional)
- **Year**: Publication year (optional)
- **Author** or **Authors**: Author names (optional)
- **DOI**: DOI identifier (optional)

### Output: abstract_units.jsonl

Each line is a JSON object representing one abstract:

```json
{
  "id": "a1b2c3d4e5f6g7h8_abstract",
  "text": "This paper presents a novel approach to...",
  "metadata": {
    "source_filename": "High-temperature Properties of Superalloys",
    "type": "abstract",
    "source_journal": "Materials Science and Engineering",
    "year": "2023",
    "author": "Smith, J. et al.",
    "DOI": "10.1016/j.msea.2023.12345"
  }
}
```

- **id**: MD5 hash of title (first 16 chars) + "_abstract"
- **text**: The abstract content
- **metadata**: Includes all available fields from Excel

### Output: extracted_abstract_graph.jsonl

Each line contains extracted graph data:

```json
{
  "id": "a1b2c3d4e5f6g7h8_abstract",
  "graph": {
    "entities": [
      {
        "id": "e-1",
        "name": "Inconel 718",
        "type": "Material",
        "description": "Nickel-based superalloy",
        "attributes": {
          "composition": "Ni-Cr-Fe alloy"
        }
      }
    ],
    "relationships": [
      {
        "id": "r-1",
        "source": "e-1",
        "target": "e-2",
        "relationship": "EXHIBITS",
        "description": "Material shows property",
        "weight": 5,
        "source_sentence": "Inconel 718 exhibits excellent..."
      }
    ]
  }
}
```

## Configuration

Add to `config/settings.yaml`:

```yaml
abstract_extraction:
  input_excel: "data/papers/superalloy_research.xlsx"  # Excel file path
  output_dir: "data/chunks"  # abstract_units.jsonl location
  output_filename: "abstract_units.jsonl"
  graph_output_dir: "data/graphs/extracted"
  graph_output_filename: "extracted_abstract_graph.jsonl"
  requests_dir: "data/cache"
  requests_filename: "extraction_abstract_requests.jsonl"
```

## Usage

### Basic Execution

```bash
# Set API key
$env:QWEN_API_KEY="your-api-key"

# Run extraction
cd D:\Pycharm\Projects\SuperalloyKgRAG
python core/pipeline_qwen/abstract_extraction.py
```

### Step-by-Step Process

1. **Reads Excel file**: Loads abstracts and metadata
2. **Generates abstract_units.jsonl**: Creates intermediate chunk file
3. **Prepares batch requests**: Creates API request file with text_to_graph prompt
4. **Uploads to Qwen API**: Submits batch job
5. **Polls for completion**: Monitors job status
6. **Downloads results**: Retrieves extracted graph data
7. **Saves output**: Writes to `extracted_abstract_graph.jsonl`

## Dependencies

```bash
pip install pandas openpyxl pyyaml openai
```

## Integration with Graph Builder

To merge abstract-extracted graphs with main graph:

```python
from core.pipeline_qwen.graph_builder_qwen import GraphBuilder

builder = GraphBuilder()
builder.merge_graphs([
    "data/graphs/extracted/extracted_graph.jsonl",
    "data/graphs/extracted/extracted_table_graph.jsonl",
    "data/graphs/extracted/extracted_abstract_graph.jsonl"
])
```

## Error Handling

- **Missing Abstract Column**: Raises ValueError with available columns list
- **Empty Abstracts**: Logs warning and skips row
- **Missing Title**: Uses row index as fallback identifier
- **Missing Optional Columns**: Gracefully omits from metadata
- **API Errors**: Detailed logging of batch job status and errors

## Column Detection

The module uses flexible column name matching:
- Searches for substrings (case-insensitive)
- "Title" matches: "Title", "Paper Title", "Article Title"
- "Abstract" matches: "Abstract", "Summary", "Abs"
- "Journal" matches: "Journal", "Source", "Publication"
- etc.

## Logging

Logs are written to `logs/superalloyKgRAG.log`:

```
2025-12-08 10:00:00 - INFO - 正在从 data/papers/superalloy_research.xlsx 读取摘要...
2025-12-08 10:00:01 - INFO - 找到的列映射: {'title': 'Title', 'abstract': 'Abstract', ...}
2025-12-08 10:00:02 - INFO - ✅ 成功生成 abstract_units.jsonl，共 150 条摘要记录
2025-12-08 10:00:05 - INFO - ✅ 批量作业已创建: batch_abc123
2025-12-08 10:15:00 - INFO - ✅ 作业成功完成！
2025-12-08 10:15:05 - INFO - 🎉 结果处理完成！成功处理 148 条，失败 2 条
```

## Comparison with Other Extractors

| Feature | extraction_qwen | table_extraction | abstract_extraction |
|---------|----------------|------------------|---------------------|
| Input | text_units.jsonl | VLM parsed JSONs | Excel files |
| Prompt | text_to_graph.md | table_to_graph.md | text_to_graph.md |
| Source | PDF text chunks | Tables from PDFs | Research abstracts |
| Metadata | PDF filename | Table caption | Title, Journal, etc. |

## Troubleshooting

### "Excel 文件缺少 'Abstract' 列"
- Check Excel column names
- Ensure at least one column contains "abstract" (case-insensitive)

### "pandas not available"
```bash
pip install pandas openpyxl
```

### Batch job stuck in "processing"
- Normal for large files (can take hours)
- Check `sleep_interval` in settings (default: 60 seconds)
- Monitor `logs/superalloyKgRAG.log` for updates

### Empty abstract_units.jsonl
- Check if Excel file has data
- Verify Abstract column is not empty
- Check logs for specific error messages

## Future Enhancements

- [ ] Support multiple Excel sheets
- [ ] CSV input support
- [ ] Custom column mapping configuration
- [ ] Abstract preprocessing (cleaning, normalization)
- [ ] Parallel processing for large datasets
- [ ] Resume capability for interrupted jobs

