# 富化提取模块 (Enrich Extraction) 实现文档

## 概述

本文档描述了富化提取模块的实现，该模块检查并合并四种类型的知识图谱：
1. **纯文本图谱** (`extracted_graph.jsonl`)
2. **摘要图谱** (`extracted_abstract_graph.jsonl`)
3. **图片图谱** (`extracted_image_graph.jsonl`)
4. **表格图谱** (`extracted_table_graph.jsonl`)

## 核心功能

### 1. 图谱检查
自动检查 `data/graphs/extracted/` 目录下是否存在四种图谱文件，并验证文件是否非空。

### 2. 自动提取
如果发现缺失的图谱类型，自动执行对应的提取流程：
- **纯文本图谱**: 执行 `loader` + `extraction_qwen`
- **摘要图谱**: 执行 `abstract_extraction`
- **图片图谱**: 执行 `image_extraction`
- **表格图谱**: 执行 `table_extraction`

### 3. 图谱合并
将所有可用的图谱合并为一个统一的 `enriched_graph.jsonl` 文件，存放在 `data/graphs/enriched/` 目录。

## 文件结构

```
core/pipeline_qwen/
├── enrich_extraction.py       # 富化提取主模块
├── loader.py                  # 文本分块模块
├── extraction_qwen.py         # 纯文本三元组提取
├── abstract_extraction.py     # 摘要图谱提取
├── image_extraction.py        # 图片图谱提取
└── table_extraction.py        # 表格图谱提取

data/graphs/
├── extracted/                 # 原始提取的图谱
│   ├── extracted_graph.jsonl
│   ├── extracted_abstract_graph.jsonl
│   ├── extracted_image_graph.jsonl
│   └── extracted_table_graph.jsonl
└── enriched/                  # 富化后的图谱
    └── enriched_graph.jsonl
```

## 流水线变更

### 原有流水线（5步）
1. OCR解析 (PDF → JSON)
2. 文本分块 (JSON → Text Units)
3. 三元组提取 (Text Units → Graph Triples)
4. 图谱构建 (消歧 → 合并 → 社区发现)
5. 向量化存储 (Graph → Vector DB)

### 新流水线（4步）
1. OCR解析 (PDF → JSON)
2. **富化提取 (文本+摘要+图片+表格 → Enriched Graph)** ⭐ 新增
3. 图谱构建 (消歧 → 合并 → 社区发现)
4. 向量化存储 (Graph → Vector DB)

## 配置文件更新

### settings.yaml 新增配置

```yaml
enrich_extraction:
  extracted_dir: "data/graphs/extracted"  # 提取的图谱存放目录
  enriched_dir: "data/graphs/enriched"    # 富化后的图谱存放目录
  enriched_filename: "enriched_graph.jsonl"  # 富化图谱文件名

graph_builder:
  input_path: "data/graphs/enriched/enriched_graph.jsonl"  # 更新为使用富化图谱
  # ...其他配置保持不变
```

## 使用方法

### 1. 直接运行富化提取模块

```bash
cd D:\Pycharm\Projects\SuperalloyKgRAG
python core/pipeline_qwen/enrich_extraction.py
```

### 2. 在流水线中使用

```bash
# 执行完整流水线（步骤1-4）
python app/run_index_qwen.py

# 仅执行步骤2（富化提取）
python app/run_index_qwen.py --step 2

# 从步骤2开始执行
python app/run_index_qwen.py --start 2

# 执行步骤2-3
python app/run_index_qwen.py --start 2 --end 3
```

### 3. 在代码中调用

```python
from core.pipeline_qwen.enrich_extraction import run_enrich_extraction

# 执行富化提取
enriched_graph_path = run_enrich_extraction()
print(f"富化图谱路径: {enriched_graph_path}")
```

## 核心类说明

### GraphType
定义四种图谱类型的元数据：

```python
class GraphType:
    TEXT = {
        'name': '纯文本图谱',
        'filename': 'extracted_graph.jsonl',
        'extract_func': 'extract_text_graph',
        'config_key': 'extraction'
    }
    # ...其他图谱类型定义
```

### EnrichExtractor
富化提取器主类，负责：
- 检查图谱存在性
- 执行缺失图谱的提取
- 合并所有图谱

主要方法：
- `check_graph_exists(graph_type)`: 检查指定类型的图谱是否存在
- `extract_text_graph()`: 提取纯文本图谱
- `extract_abstract_graph()`: 提取摘要图谱
- `extract_image_graph()`: 提取图片图谱
- `extract_table_graph()`: 提取表格图谱
- `check_and_extract_all()`: 检查并提取所有缺失的图谱
- `merge_graphs()`: 合并所有图谱
- `run()`: 执行完整的富化提取流程

## 执行流程

```
开始
  ↓
检查 extracted_graph.jsonl
  ├─ 存在 → 跳过
  └─ 不存在 → 执行 loader + extraction_qwen
  ↓
检查 extracted_abstract_graph.jsonl
  ├─ 存在 → 跳过
  └─ 不存在 → 执行 abstract_extraction
  ↓
检查 extracted_image_graph.jsonl
  ├─ 存在 → 跳过
  └─ 不存在 → 执行 image_extraction
  ↓
检查 extracted_table_graph.jsonl
  ├─ 存在 → 跳过
  └─ 不存在 → 执行 table_extraction
  ↓
合并所有图谱为 enriched_graph.jsonl
  ↓
输出到 data/graphs/enriched/
  ↓
结束
```

## 输出格式

### enriched_graph.jsonl 格式

每行包含一个图谱条目，带有来源类型标记：

```json
{
  "id": "chunk-xxxx",
  "graph": {
    "entities": [...],
    "relationships": [...],
    "source_type": "纯文本图谱"  // 新增字段，标识来源
  }
}
```

## 日志输出示例

```
================================================================================
富化提取流程启动
================================================================================
提取图谱目录: D:\Pycharm\Projects\SuperalloyKgRAG\data\graphs\extracted
富化图谱目录: D:\Pycharm\Projects\SuperalloyKgRAG\data\graphs\enriched

================================================================================
📊 开始检查四种图谱...
================================================================================
✅ 纯文本图谱 已存在: D:\...\extracted_graph.jsonl
✅ 摘要图谱 已存在: D:\...\extracted_abstract_graph.jsonl
✅ 图片图谱 已存在: D:\...\extracted_image_graph.jsonl
✅ 表格图谱 已存在: D:\...\extracted_table_graph.jsonl

================================================================================
📊 图谱检查结果:
  纯文本图谱: ✅ 存在
  摘要图谱: ✅ 存在
  图片图谱: ✅ 存在
  表格图谱: ✅ 存在
================================================================================

================================================================================
🔗 开始合并图谱...
================================================================================
📥 合并 纯文本图谱...
  ✅ 纯文本图谱: 合并 1234 条记录
📥 合并 摘要图谱...
  ✅ 摘要图谱: 合并 567 条记录
📥 合并 图片图谱...
  ✅ 图片图谱: 合并 890 条记录
📥 合并 表格图谱...
  ✅ 表格图谱: 合并 345 条记录

================================================================================
✅ 图谱合并完成！
📊 总计合并: 3036 条记录
📁 输出文件: D:\...\enriched_graph.jsonl
================================================================================

🎉 富化提取流程完成！
```

## 错误处理

模块包含完善的错误处理机制：

1. **文件不存在**: 自动触发对应的提取流程
2. **文件为空**: 记录警告并标记为不存在
3. **提取失败**: 记录错误日志，继续处理其他图谱
4. **合并失败**: 记录错误并抛出异常

## 测试

运行测试脚本验证功能：

```bash
python tests/test_enrich_extraction.py
```

测试内容包括：
1. 图谱类型定义测试
2. 配置加载测试
3. 提取器初始化测试
4. 图谱存在性检查测试
5. Graph Builder 输入路径配置测试

## 优势

1. **自动化**: 自动检测和补全缺失的图谱
2. **灵活性**: 支持增量更新，只提取缺失的部分
3. **统一性**: 将多源图谱统一到一个文件中
4. **可追溯**: 每个图谱条目标记了来源类型
5. **简化流程**: 将原有5步流程简化为4步

## 注意事项

1. 确保 `data/processed_jsons/` 目录包含 VLM 解析后的 JSON 文件
2. 摘要提取需要 `data/papers/superalloy_research.xlsx` 文件
3. 各提取模块需要有效的 API 密钥配置
4. 批量处理可能需要较长时间，建议使用批处理模式

## 相关文件

- `core/pipeline_qwen/enrich_extraction.py` - 主模块实现
- `app/run_index_qwen.py` - 流水线执行脚本
- `config/settings.yaml` - 配置文件
- `tests/test_enrich_extraction.py` - 测试脚本
- `docs/ENRICH_EXTRACTION_GUIDE.md` - 本文档

## 更新历史

- 2025-12-09: 初始实现，替换原有的 loader + extraction 步骤

