# ontology.py 快速入门

## 🚀 快速开始

### 1️⃣ 首次使用（完整分析）
```bash
cd draw
python ontology.py --mode full
```

### 2️⃣ 快速重绘图表
```bash
python ontology.py --mode viz
```

## 📋 三种模式

| 模式 | 命令 | 用途 | 速度 |
|------|------|------|------|
| **full** | `--mode full` | 完整流程（分析+可视化） | 慢 ⏱️ |
| **viz** | `--mode viz` | 仅重绘图表（使用缓存） | 快 ⚡ |
| **analyze** | `--mode analyze` | 仅分析（不绘图） | 中速 |

## 🎨 典型工作流程

### 初次分析
```bash
# 1. 完整分析
python ontology.py --mode full

# 2. 查看结果
# - visualizations/relationship_matrix.svg
# - visualizations/relationship_types_distribution.svg
# - visualizations/superalloy_relationships.xlsx
```

### 调整图表样式
```bash
# 1. 修改 visualize_relationship_matrix() 函数
# 例如：改变颜色、字体、大小等

# 2. 快速重绘（无需重新分析）
python ontology.py --mode viz

# 3. 查看新图表
# 如果不满意，继续修改并重复步骤2
```

## 📂 输出文件

```
visualizations/
├── relationship_matrix.svg           # 关系矩阵热图（矢量图）
├── relationship_matrix.png           # 关系矩阵热图（位图备份）
├── relationship_types_distribution.svg  # 关系类型分布
├── relationship_types_distribution.png
├── superalloy_relationships.xlsx     # 详细数据表格
└── analysis_results.pkl              # 分析结果缓存
```

## ⚙️ 高级选项

### 自定义路径
```bash
python ontology.py \
  --graph-path /path/to/graph.json \
  --output-dir /path/to/output \
  --cache-file my_cache.pkl
```

### 查看帮助
```bash
python ontology.py --help
```

## 💡 提示

- ✅ 使用 `--mode viz` 可以节省大量时间
- ✅ SVG 格式适合论文发表（矢量图，可无损缩放）
- ✅ PNG 格式适合 PPT 演示
- ✅ Excel 文件包含所有详细数据

## 🔍 数据分析

查看 `superalloy_relationships.xlsx` 包含：
- **总览** sheet：所有类别对的关系统计
- **详细关系** sheets：每对类别的具体关系
- **关系类型统计** sheet：所有关系类型的频次

## 📊 关系矩阵说明

- **行**：源类别 (Source Category)
- **列**：目标类别 (Target Category)
- **数值**：从源到目标的关系数量
- **非对称**：matrix[i][j] ≠ matrix[j][i]（区分方向）

## ❓ 常见问题

### Q: 为什么 viz 模式报错？
A: 请先运行 `--mode full` 生成缓存文件

### Q: 如何修改图表颜色？
A: 编辑 `visualize_relationship_matrix()` 函数中的 colormap

### Q: 如何添加新的可视化？
A: 在类中添加新方法，然后在 main() 中调用

## 📚 更多信息

详细文档：`USAGE.md`

