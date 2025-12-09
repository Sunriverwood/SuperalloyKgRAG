# Image Extraction Module Documentation

## 概述

图片提取模块（`image_extraction.py`）仿照表格提取方法，对不同类型的图片提取知识图谱三元组，主要从图片的 **caption**、**description**、**trend**、**summary** 等方面提取结构化信息。

## 功能特性

### 1. 多类型图片支持

支持以下四种图片类型的专业化提取：

- **chart（图表）**：折线图、柱状图、散点图等
  - 提取要素：标题、坐标轴、图例、趋势描述、数据点
  - Prompt: `config/prompts/chart_to_graph.md`

- **schematic（示意图/图解）**：结构图、流程图、机制图
  - 提取要素：组件、结构、流程、空间关系、OCR文字
  - Prompt: `config/prompts/schematic_to_graph.md`

- **photograph（显微照片/实物图）**：SEM、TEM、光学显微镜等
  - 提取要素：相位、缺陷、特征、形貌、分布
  - Prompt: `config/prompts/photograph_to_graph.md`

- **other（其他）**：默认使用 photograph 的提取策略

### 2. 知识图谱提取

每种图片类型都有专门的知识图谱提取策略：

#### Chart（图表）提取示例
```json
{
  "entities": [
    {
      "id": "e-1",
      "name": "Alloy 718",
      "type": "MATERIAL",
      "description": "Alloy 718 exhibits temperature-dependent yield strength...",
      "attributes": {"source_chart": "page_5_block_3"}
    },
    {
      "id": "e-2",
      "name": "Yield Strength",
      "type": "PROPERTY",
      "description": "Mechanical property measuring material strength...",
      "attributes": {"unit": "MPa", "value_range": "800-1200"}
    }
  ],
  "relationships": [
    {
      "id": "r-1",
      "source": "e-temperature",
      "target": "e-yield-strength",
      "relationship": "DECREASES",
      "description": "Increasing temperature causes yield strength to decrease",
      "attributes": {"trend_type": "decreasing"}
    }
  ]
}
```

#### Schematic（示意图）提取示例
```json
{
  "entities": [
    {
      "id": "e-2",
      "name": "Base Metal",
      "type": "REGION",
      "description": "Unaffected base metal region in welded structure.",
      "attributes": {"ocr_labels": "Base Metal", "position": "left"}
    },
    {
      "id": "e-3",
      "name": "Heat-Affected Zone",
      "type": "REGION",
      "description": "HAZ region showing grain structure modifications..."
    }
  ],
  "relationships": [
    {
      "id": "r-1",
      "source": "e-base-metal",
      "target": "e-weld",
      "relationship": "PART_OF",
      "description": "Base metal is part of the welded structure."
    }
  ]
}
```

#### Photograph（显微照片）提取示例
```json
{
  "entities": [
    {
      "id": "e-2",
      "name": "δ-phase",
      "type": "PHASE",
      "description": "Delta phase precipitates with needle-shaped morphology...",
      "attributes": {
        "morphology": "needle-shaped",
        "size": "2-5 μm length",
        "distribution": "grain boundaries"
      }
    }
  ],
  "relationships": [
    {
      "id": "r-3",
      "source": "e-delta-phase",
      "target": "e-grain-boundary",
      "relationship": "FORMS_AT",
      "description": "δ-phase precipitates form at grain boundary locations."
    }
  ]
}
```

## 工作流程

### 1. 输入数据格式

从 VLM 解析后的 JSON 文件中提取图片信息，期望的 block 格式：

```json
{
  "block_id": "page_5_block_3",
  "type": "image",
  "caption": "Effect of temperature on yield strength",
  "image_type": "chart",
  "content": {
    "title": "Temperature vs. Yield Strength",
    "x_axis_label": "Temperature (°C)",
    "y_axis_label": "Yield Strength (MPa)",
    "legend": ["Alloy 718", "Alloy 625"],
    "trend_description": "Yield strength decreases from 1200 MPa at 20°C to 800 MPa at 600°C",
    "extracted_data": [{"x": 20, "y": 1200}, {"x": 600, "y": 800}]
  }
}
```

### 2. 处理流程

```
1. prepare_batch_requests()
   ├─ 扫描 data/processed_jsons/ 目录
   ├─ 提取所有 type="image" 的 blocks
   ├─ 根据 image_type 选择对应的 prompt
   ├─ 生成批量请求文件（extraction_image_requests.jsonl）
   └─ 生成图片 units 文件（image_units.jsonl）

2. 上传批量请求 → Qwen Batch API

3. 轮询作业状态（每 60 秒检查一次）

4. 处理结果
   ├─ 下载 LLM 返回的图谱数据
   ├─ 解析 JSON 格式的三元组
   └─ 保存到 extracted_image_graph.jsonl
```

### 3. 输出文件

#### a. image_units.jsonl（图片 text units）
与 loader 输出格式一致，用于向量检索：

```json
{
  "id": "chunk-abc123...",
  "document_id": "doc-def456...",
  "text": "Caption: Effect of temperature...\nTitle: Temperature vs. Yield Strength...\nTrend: Yield strength decreases...\nFull Content: {...}",
  "metadata": {
    "source_filename": "paper1.json",
    "pages": [5],
    "blocks": ["page_5_block_3"],
    "image_type": "chart"
  }
}
```

#### b. extracted_image_graph.jsonl（图片知识图谱）
```json
{
  "id": "chunk-abc123...",
  "graph": {
    "entities": [...],
    "relationships": [...]
  }
}
```

## 配置说明

在 `config/settings.yaml` 中添加：

```yaml
image_extraction:
  input_dir: "data/processed_jsons"
  output_dir: "data/chunks"
  output_filename: "image_units.jsonl"
  graph_output_dir: "data/graphs/extracted"
  graph_output_filename: "extracted_image_graph.jsonl"
  requests_dir: "data/cache"
  requests_filename: "extraction_image_requests.jsonl"
```

## 使用方法

### 独立运行

```bash
# 确保已设置 QWEN_API_KEY 环境变量
export QWEN_API_KEY="your-api-key"

# 运行图片提取
python core/pipeline_qwen/image_extraction.py
```

### 在代码中使用

```python
from core.pipeline_qwen.image_extraction import ImageProcessor

# 初始化处理器
processor = ImageProcessor(config_path="config/settings.yaml")

# 仅准备批量请求（不执行）
image_count = processor.prepare_batch_requests()
print(f"准备了 {image_count} 个图片的提取请求")

# 完整执行流程
processor.run()
```

## Prompt 设计原则

每个图片类型的 prompt 都遵循以下原则：

1. **明确角色定位**：KG Builder for specific image type
2. **清晰的规则**：
   - 实体识别规则（类型、属性）
   - 关系提取规则（关系类型、方向）
   - 上下文化要求（单位、缩写解析）
3. **严格的输出格式**：JSON Schema
4. **领域知识指导**：针对材料科学的专业要求
   - Chart: 趋势分析、数据点关联
   - Schematic: 结构关系、组件功能
   - Photograph: 形貌特征、相位分布

## 与表格提取的对比

| 特性 | Table Extraction | Image Extraction |
|------|------------------|------------------|
| 输入类型 | 表格数据（data 数组） | 图片元数据（content 对象） |
| Prompt 数量 | 1 个通用 prompt | 3 个专用 prompt |
| 上下文构建 | 直接使用 JSON | 根据类型提取不同字段 |
| 实体类型 | 材料、属性、成分 | 相位、缺陷、趋势、特征 |
| 关系类型 | HAS_PROPERTY | SHOWS_TREND、FORMS_AT、PART_OF |

## 批量处理优势

1. **成本效率**：批量 API 价格为标准价格的 50%
2. **并发处理**：一次性提交所有图片的提取任务
3. **容错性**：单个图片失败不影响整体流程
4. **可追踪**：每个图片有唯一的 chunk_id

## 日志示例

```
2025-12-08 10:00:00 - INFO - 开始准备图片批量请求...
2025-12-08 10:00:01 - INFO - 找到 15 个 JSON 文件
2025-12-08 10:00:05 - INFO - ✅ 成功创建批量请求文件，共 45 个图片
2025-12-08 10:00:05 - INFO - ✅ 成功创建 image units 文件
2025-12-08 10:00:06 - INFO - 📤 正在上传批量请求文件...
2025-12-08 10:00:08 - INFO - ✅ 文件上传成功: file-abc123
2025-12-08 10:00:09 - INFO - 🚀 正在创建批量作业...
2025-12-08 10:00:10 - INFO - ✅ 批量作业已创建: batch-def456
2025-12-08 10:00:11 - INFO - ⏳ 开始轮询作业状态...
2025-12-08 10:01:11 - INFO -   - 当前状态: in_progress
...
2025-12-08 10:15:00 - INFO -   - 当前状态: completed
2025-12-08 10:15:01 - INFO - 📥 正在下载结果文件...
2025-12-08 10:15:05 - INFO - 🎉 结果处理完成！成功处理 43 个图片，失败 2 个。
2025-12-08 10:15:05 - INFO - 💾 Image units 已保存至: data/chunks/image_units.jsonl
2025-12-08 10:15:05 - INFO - 💾 图谱数据已保存至: data/graphs/extracted/extracted_image_graph.jsonl
```

## 后续集成

提取的图片知识图谱可以与其他模块集成：

1. **图谱合并**：与文本、表格提取的图谱合并
   ```python
   # 在 graph_builder 中添加
   image_graph = load_jsonl("data/graphs/extracted/extracted_image_graph.jsonl")
   ```

2. **向量嵌入**：使用 image_units.jsonl 进行嵌入
   ```python
   # 在 embedding 中添加
   image_units = load_jsonl("data/chunks/image_units.jsonl")
   ```

3. **混合检索**：结合文本、表格、图片的检索结果
   ```python
   # 在 query 中扩展检索源
   results = search_text_units() + search_table_units() + search_image_units()
   ```

## 常见问题

### 1. 如果 prompt 文件加载失败？
模块会回退到默认的通用 prompt，并记录警告日志。

### 2. 如何添加新的图片类型？
1. 在 `IMAGE_TYPE_PROMPTS` 中添加映射
2. 创建对应的 prompt 文件
3. 在 `_build_image_context()` 中添加上下文构建逻辑

### 3. 批量作业失败怎么办？
检查日志中的错误详情，常见原因：
- API 配额不足
- 请求格式错误
- 网络连接问题

可以重新运行 `processor.run()`，会重新创建批量请求。

## 性能优化建议

1. **分批处理**：如果图片数量超过 1000 个，建议分批处理
2. **缓存结果**：已处理的图片不会重复处理（通过 chunk_id 去重）
3. **并行处理**：可以同时运行表格和图片的提取流程

## 版本历史

- v1.0 (2025-12-08): 初始版本，支持 chart、schematic、photograph 三种类型

