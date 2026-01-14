# 知识图谱可视化文档

## 概述

本模块提供知识图谱拓扑结构的统计分析和可视化功能，使用 `scienceplots` 库生成高质量的学术风格图表。

## 功能特性

### 1. **度分布图** (`degree_distribution.svg`)
- **线性坐标**: 展示节点度的直方图分布
- **对数坐标**: Log-Log图，用于识别幂律分布特征
- **用途**: 分析图谱的度中心性，识别枢纽节点

### 2. **节点类型分布** (`node_type_distribution.svg`)
- **饼图**: 展示Top 15节点类型的占比
- **用途**: 了解图谱中实体类型的多样性和主要构成

### 3. **关系类型分布** (`relationship_type_distribution.svg`)
- **水平条形图**: 展示Top 20关系类型及其频次
- **用途**: 分析实体间的主要连接模式

### 4. **社区分布** (`community_distribution.svg`)
- **社区大小直方图**: 展示社区规模分布
- **累积分布曲线**: 展示节点在社区中的累积分布
- **用途**: 评估图谱的模块化结构和社区检测效果

### 5. **中心性对比** (`centrality_comparison.svg`)
- **介数中心性**: Top 30 最重要的桥接节点
- **紧密中心性**: Top 30 最接近其他节点的实体
- **注意**: 仅对节点数 < 10,000 的图谱计算（性能优化）

### 6. **图连通性分析** (`graph_connectivity.svg`)
- **连通分量大小**: 展示各连通分量的规模
- **度分布**: 从连通性角度分析节点度
- **用途**: 评估图谱的整体连通性和碎片化程度

### 7. **边权重分布** (`edge_weight_distribution.svg`)
- **直方图**: 边权重的频次分布
- **累积分布**: 边权重的累积概率分布
- **注意**: 仅当图谱包含边权重时生成

### 8. **Schema热力图** (`schema_heatmap.svg`)
- **实体类型关系矩阵**: 展示Top 15实体类型之间的关系数量
- **用途**: 分析图谱的Schema结构，识别核心实体类型间的联系

### 9. **统计摘要** (`statistics_summary.txt`)
- 基本信息（节点数、边数、平均度等）
- 度统计（最大度、最小度、中位数）
- 连通性分析
- 节点类型统计
- 社区统计

## 使用方法

### 基本用法

```bash
cd draw
python statistics.py
```

### 编程调用

```python
from draw.statistics import GraphStatistics

# 创建统计分析器
stats = GraphStatistics(
    graph_path="data/graphs/final_graph.json",
    output_dir="visualizations"
)

# 运行所有可视化
stats.run_all_visualizations()

# 或单独运行某个可视化
stats.plot_degree_distribution()
stats.plot_schema_heatmap()
```

## 性能优化

### 1. **缓存优化**
- 使用 `@lru_cache` 缓存节点度量计算结果
- 避免重复计算中心性指标

### 2. **条件计算**
- 图谱规模 >= 10,000 节点时跳过中心性计算
- 大图谱只计算基础统计指标

### 3. **高效数据结构**
- 使用 `Counter` 和 `defaultdict` 优化统计计算
- 使用 `nx.node_link_graph` 高效加载图谱

### 4. **并行处理支持**
- 预留 `ProcessPoolExecutor` 和 `ThreadPoolExecutor` 接口
- 可扩展支持多图谱批量处理

## 输出格式

- **优先格式**: SVG（矢量图，可无损缩放）
- **备份格式**: PNG（300 DPI）
- **文本输出**: UTF-8编码的统计摘要

## 图表风格

使用 `scienceplots` 库的学术风格：
- 清晰的网格线
- 专业的配色方案
- 适合论文发表的高质量输出

## 依赖项

```bash
pip install matplotlib scienceplots seaborn networkx numpy
```

## 目录结构

```
SuperalloyKgRAG/
├── draw/
│   └── statistics.py          # 可视化脚本
├── visualizations/             # 输出目录
│   ├── degree_distribution.svg
│   ├── node_type_distribution.svg
│   ├── relationship_type_distribution.svg
│   ├── community_distribution.svg
│   ├── centrality_comparison.svg (小图谱)
│   ├── graph_connectivity.svg
│   ├── edge_weight_distribution.svg (如果有权重)
│   ├── schema_heatmap.svg
│   └── statistics_summary.txt
└── data/
    └── graphs/
        └── final_graph.json    # 输入图谱
```

## 示例输出

### 统计摘要
```
==================================================
知识图谱统计摘要
==================================================

基本信息:
  节点数: 52783
  边数: 79550
  是否有向图: True
  平均度: 3.01

度统计:
  最大度: 1143
  最小度: 0
  度中位数: 2.00

连通性:
  弱连通分量数: 2254
  最大弱连通分量大小: 46061

节点类型统计:
  节点类型数: 5328
  最常见类型: [('Material', 3927), ('Microstructural Feature', 3493), ...]

社区统计:
  社区数: 2336
  平均社区大小: 22.60
```

## 自定义配置

### 修改图表风格
```python
# 在脚本开头修改
plt.style.use(['science', 'ieee'])  # IEEE风格
plt.style.use(['science', 'nature'])  # Nature风格
```

### 修改输出格式
```python
stats = GraphStatistics(graph_path, output_dir)
stats.image_format = 'png'  # 改为PNG
stats.dpi = 600  # 提高分辨率
```

### 调整性能阈值
```python
# 在 _compute_node_metrics 中修改
if self.G.number_of_nodes() < 20000:  # 提高阈值
    # 计算中心性
```

## 注意事项

1. **内存使用**: 大图谱（>100k节点）可能需要较大内存
2. **计算时间**: 中心性计算在大图谱上很慢，已设置阈值跳过
3. **编码**: 确保终端支持UTF-8以正确显示日志信息
4. **文件覆盖**: 重复运行会覆盖已有可视化文件

## 故障排除

### 问题: "No module named 'scienceplots'"
**解决**: 
```bash
pip install scienceplots
```

### 问题: SVG文件无法打开
**解决**: 使用现代浏览器（Chrome/Firefox）或专业软件（Inkscape/Adobe Illustrator）

### 问题: 中文乱码
**解决**: 
```python
# 在脚本中添加
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
matplotlib.rcParams['axes.unicode_minus'] = False
```

## 未来扩展

- [ ] 添加时间序列分析（图谱演化）
- [ ] 支持交互式可视化（Plotly）
- [ ] 自动生成可视化报告（PDF/HTML）
- [ ] 支持多图谱对比分析
- [ ] 添加更多拓扑指标（聚类系数、直径等）

## 许可证

Apache License 2.0 - 详见项目根目录LICENSE文件

---

**作者**: SUNRIVERWOOD  
**最后更新**: 2025-12-18

