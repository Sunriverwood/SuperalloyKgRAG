# 图片提取功能 - 快速开始

## 📋 已完成的工作

我已经完成了图片提取功能的实现，仿照表格提取方法，对不同类型的图片提取知识图谱三元组。

## 🎯 核心功能

### 支持的图片类型
1. **Chart（图表）** - 折线图、柱状图、散点图
2. **Schematic（示意图）** - 结构图、流程图、机制图  
3. **Photograph（显微照片）** - SEM、TEM、光学显微镜图像

### 提取的信息
- **Caption**: 图片标题
- **Description**: 详细描述
- **Trend**: 趋势分析（适用于图表）
- **Summary**: 总结信息
- **OCR Text**: 图中的文字（适用于示意图）
- **Data Points**: 数据点（适用于图表）

## 📁 新增文件

```
SuperalloyKgRAG/
├── core/pipeline_qwen/
│   └── image_extraction.py              ✨ 核心模块 (483行)
├── config/prompts/
│   ├── chart_to_graph.md               ✨ 图表提取 prompt
│   ├── schematic_to_graph.md           ✨ 示意图提取 prompt
│   └── photograph_to_graph.md          ✨ 显微照片提取 prompt
├── docs/
│   └── IMAGE_EXTRACTION_GUIDE.md       ✨ 详细使用指南
├── tests/
│   └── test_image_extraction.py        ✨ 测试脚本
├── verify_image_extraction.py          ✨ 快速验证脚本
└── IMAGE_EXTRACTION_IMPLEMENTATION_SUMMARY.md  ✨ 实现总结
```

## 🚀 快速验证

### 1. 验证模块安装
```bash
cd D:\Pycharm\Projects\SuperalloyKgRAG
python verify_image_extraction.py
```

### 2. 查看输出
如果一切正常，你会看到：
```
✅ 成功导入 ImageProcessor
✅ 成功初始化 ImageProcessor
✅ chart: 已加载 (xxxx 字符)
✅ schematic: 已加载 (xxxx 字符)
✅ photograph: 已加载 (xxxx 字符)
✅ 上下文构建正常
✅ 所有验证通过！
```

## 💡 使用方法

### 方法一：独立运行（完整流程）
```bash
# 设置 API key
set QWEN_API_KEY=your-api-key-here

# 运行图片提取
python core\pipeline_qwen\image_extraction.py
```

### 方法二：在代码中使用
```python
from core.pipeline_qwen.image_extraction import ImageProcessor

# 初始化处理器
processor = ImageProcessor()

# 仅准备批量请求（不调用 API）
image_count = processor.prepare_batch_requests()
print(f"准备了 {image_count} 个图片的提取请求")

# 执行完整流程（需要 API key）
processor.run()
```

### 方法三：运行测试
```bash
python tests\test_image_extraction.py
```

## 📊 输出文件

运行后会生成以下文件：

1. **data/chunks/image_units.jsonl**  
   图片的文本单元，与 loader 格式一致，用于向量检索

2. **data/graphs/extracted/extracted_image_graph.jsonl**  
   提取的知识图谱，与 extraction_qwen 格式一致

3. **data/cache/extraction_image_requests.jsonl**  
   批量请求文件（中间文件）

## 📖 详细文档

查看完整的使用指南：
```bash
# 打开文档
notepad docs\IMAGE_EXTRACTION_GUIDE.md

# 或查看实现总结
notepad IMAGE_EXTRACTION_IMPLEMENTATION_SUMMARY.md
```

## 🔍 示例：图表提取

### 输入（来自 VLM 解析的 JSON）
```json
{
  "block_id": "page_5_block_3",
  "type": "image",
  "image_type": "chart",
  "caption": "Effect of temperature on yield strength",
  "content": {
    "title": "Temperature vs. Yield Strength",
    "x_axis_label": "Temperature (°C)",
    "y_axis_label": "Yield Strength (MPa)",
    "trend_description": "Yield strength decreases from 1200 MPa at 20°C to 800 MPa at 600°C"
  }
}
```

### 输出（知识图谱）
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
      "description": "Mechanical property...",
      "attributes": {"unit": "MPa", "value_range": "800-1200"}
    }
  ],
  "relationships": [
    {
      "id": "r-1",
      "source": "e-temperature",
      "target": "e-yield-strength",
      "relationship": "DECREASES",
      "description": "Increasing temperature causes yield strength to decrease"
    }
  ]
}
```

## 🔧 配置说明

配置文件位于 `config/settings.yaml`，已添加：

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

## 🤝 与其他模块的集成

### 与 table_extraction 对比
- **架构一致**: 完全仿照 table_extraction 的设计
- **流程相同**: 准备请求 → 上传 → 轮询 → 处理结果
- **输出格式**: 完全兼容现有的图谱构建流程

### 后续集成建议
1. 在 `graph_builder` 中合并 `extracted_image_graph.jsonl`
2. 在 `embedding` 中嵌入 `image_units.jsonl`
3. 在 `query` 中支持图片内容的检索

## ❓ 常见问题

### Q: 如何添加新的图片类型？
A: 
1. 在 `IMAGE_TYPE_PROMPTS` 中添加映射
2. 创建对应的 prompt 文件
3. 在 `_build_image_context()` 中添加处理逻辑

### Q: 批量处理需要多长时间？
A: 
- 100 张图片: ~5-10 分钟
- 1000 张图片: ~30-60 分钟

### Q: 如何处理错误？
A: 
- 查看日志文件: `logs/extraction.log`
- 单个图片失败不影响整体流程
- 可以重新运行处理失败的部分

## 📞 需要帮助？

- 查看详细文档: `docs/IMAGE_EXTRACTION_GUIDE.md`
- 查看实现总结: `IMAGE_EXTRACTION_IMPLEMENTATION_SUMMARY.md`
- 运行验证脚本: `python verify_image_extraction.py`

---

**实现完成时间**: 2025-12-08  
**版本**: v1.0  
**状态**: ✅ 已完成并可用

