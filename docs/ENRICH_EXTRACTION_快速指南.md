# 富化提取快速使用指南

## 功能说明

富化提取模块会自动：
1. ✅ **检查**四种图谱是否存在（纯文本、摘要、图片、表格）
2. 🔧 **补全**缺失的图谱（自动执行对应提取代码）
3. 🔗 **合并**所有图谱到 `enriched_graph.jsonl`

## 快速开始

### 方法一：执行完整流水线

```bash
cd D:\Pycharm\Projects\SuperalloyKgRAG
python app/run_index_qwen.py
```

这会执行完整的4步流程：
1. OCR解析
2. **富化提取** ← 新步骤
3. 图谱构建
4. 向量化存储

### 方法二：仅执行富化提取

```bash
python app/run_index_qwen.py --step 2
```

### 方法三：直接运行模块

```bash
python core/pipeline_qwen/enrich_extraction.py
```

## 目录结构

```
data/graphs/
├── extracted/              # 原始提取的图谱
│   ├── extracted_graph.jsonl           # ← 纯文本图谱
│   ├── extracted_abstract_graph.jsonl  # ← 摘要图谱
│   ├── extracted_image_graph.jsonl     # ← 图片图谱
│   └── extracted_table_graph.jsonl     # ← 表格图谱
│
└── enriched/               # 富化后的图谱
    └── enriched_graph.jsonl            # ← 合并后的统一图谱
```

## 四种图谱说明

| 图谱类型 | 文件名 | 数据来源 | 提取方法 |
|---------|--------|----------|---------|
| 📝 纯文本 | `extracted_graph.jsonl` | PDF正文 | `loader` + `extraction_qwen` |
| 📄 摘要 | `extracted_abstract_graph.jsonl` | Excel摘要 | `abstract_extraction` |
| 🖼️ 图片 | `extracted_image_graph.jsonl` | PDF图片 | `image_extraction` |
| 📊 表格 | `extracted_table_graph.jsonl` | PDF表格 | `table_extraction` |

## 执行逻辑

```
开始富化提取
    ↓
检查4种图谱是否存在
    ↓
┌─────────────────────┐
│ 存在？              │
├─────────────────────┤
│ ✅ 是 → 跳过        │
│ ❌ 否 → 执行提取    │
└─────────────────────┘
    ↓
合并所有可用图谱
    ↓
输出 enriched_graph.jsonl
```

## 配置要求

确保 `config/settings.yaml` 中包含：

```yaml
enrich_extraction:
  extracted_dir: "data/graphs/extracted"
  enriched_dir: "data/graphs/enriched"
  enriched_filename: "enriched_graph.jsonl"

graph_builder:
  input_path: "data/graphs/enriched/enriched_graph.jsonl"
```

## 前置条件

### 必需文件

1. **PDF 解析结果**: `data/processed_jsons/*.json`
   - 来源：步骤1 OCR解析
   - 用于：纯文本、图片、表格提取

2. **摘要 Excel**: `data/papers/superalloy_research.xlsx`
   - 用于：摘要提取

### API 配置

在环境变量中设置：
```bash
export QWEN_API_KEY=your_api_key_here
```

或在 `settings.yaml` 中配置：
```yaml
llm:
  api_key: ${QWEN_API_KEY}
```

## 常见场景

### 场景1：首次运行（无任何图谱）

```bash
python app/run_index_qwen.py --step 2
```

**会发生什么**：
1. 检测到4种图谱全部缺失
2. 依次执行：
   - 纯文本提取 (loader + extraction)
   - 摘要提取 (abstract_extraction)
   - 图片提取 (image_extraction)
   - 表格提取 (table_extraction)
3. 合并为 enriched_graph.jsonl

### 场景2：部分图谱已存在

假设已有纯文本和摘要图谱：

```bash
python app/run_index_qwen.py --step 2
```

**会发生什么**：
1. 检测到纯文本和摘要图谱 → 跳过
2. 检测到图片和表格图谱缺失 → 执行提取
3. 合并所有4种图谱

### 场景3：所有图谱已存在

```bash
python app/run_index_qwen.py --step 2
```

**会发生什么**：
1. 检测到4种图谱全部存在 → 全部跳过
2. 直接合并为 enriched_graph.jsonl

## 查看执行日志

日志文件位置：`logs/superalloyKgRAG.log`

关键日志标识：
- ✅ 表示成功
- ❌ 表示失败/缺失
- 🚀 表示开始执行
- 📊 表示统计信息
- 🔗 表示合并操作

## 验证结果

### 检查输出文件

```bash
# 查看富化图谱是否生成
ls data/graphs/enriched/

# 查看文件大小
ls -lh data/graphs/enriched/enriched_graph.jsonl

# 统计行数（记录数）
wc -l data/graphs/enriched/enriched_graph.jsonl
```

### 查看内容示例

```bash
# 查看前几行
head -n 3 data/graphs/enriched/enriched_graph.jsonl

# 使用 Python 查看
python -c "
import json
with open('data/graphs/enriched/enriched_graph.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i < 3:
            data = json.loads(line)
            print(f'记录 {i+1}: {data[\"id\"]}, 来源: {data[\"graph\"].get(\"source_type\", \"未知\")}')
"
```

## 故障排查

### 问题1：某个图谱提取失败

**症状**：日志显示 "❌ 提取 XXX图谱失败"

**解决**：
1. 检查对应的 API 密钥是否配置
2. 查看详细错误日志
3. 手动运行对应的提取模块：
   ```bash
   # 纯文本
   python core/pipeline_qwen/extraction_qwen.py
   
   # 摘要
   python core/pipeline_qwen/abstract_extraction.py
   
   # 图片
   python -c "from core.pipeline_qwen.image_extraction import ImageProcessor; ImageProcessor().run()"
   
   # 表格
   python -c "from core.pipeline_qwen.table_extraction import TableProcessor; TableProcessor().run()"
   ```

### 问题2：合并后的图谱为空

**症状**：enriched_graph.jsonl 文件为空或很小

**检查**：
1. 确认 `data/graphs/extracted/` 下有图谱文件
2. 检查原始图谱文件是否非空
3. 查看日志中的合并统计信息

### 问题3：找不到配置文件

**症状**：运行时报错 "配置文件未找到"

**解决**：
确保在项目根目录运行，或使用绝对路径：
```bash
cd D:\Pycharm\Projects\SuperalloyKgRAG
python app/run_index_qwen.py --step 2
```

## 性能说明

- **首次运行**：需要执行所有4种提取，耗时较长（几小时到几天）
- **增量运行**：只提取缺失的图谱，耗时短
- **仅合并**：如果所有图谱已存在，仅需几秒钟

## 下一步

完成富化提取后，继续执行：

```bash
# 步骤3：图谱构建
python app/run_index_qwen.py --step 3

# 步骤4：向量化存储
python app/run_index_qwen.py --step 4

# 或一次性执行剩余步骤
python app/run_index_qwen.py --start 3
```

## 相关文档

- 详细文档：`docs/ENRICH_EXTRACTION_GUIDE.md`
- 测试脚本：`tests/test_enrich_extraction.py`
- 主模块：`core/pipeline_qwen/enrich_extraction.py`
- 配置文件：`config/settings.yaml`

