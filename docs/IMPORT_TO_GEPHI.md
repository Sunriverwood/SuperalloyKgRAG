# Gephi 可视化指南

> 更新时间：2026-03-24（与当前主流程入口对齐）
> 
> 说明：本轮文档整理不包含 `draw/` 与 `visualizations/` 目录。

## 📋 概述

本文档说明如何使用 [Gephi](https://gephi.org/) 对 SuperalloyKgRAG 生成的知识图谱进行专业级可视化分析。通过 Gephi，我们可以展示图谱的宏观拓扑结构、社区聚类效果以及关键实体的重要性。

## 🛠️ 前置准备

1.  **安装 Gephi**: 请前往 [Gephi 官网](https://gephi.org/) 下载并安装最新版本 (推荐 0.9.x 或 0.10.x)。
2.  **准备数据**: 运行app/gephi.py，将`final_graph.json`转为 `.gexf` 格式的图谱文件（例如 `final_graph.gexf`）。
    * *注：Gephi 虽然支持 JSON，但 GEXF 格式能更好地保留节点属性（如社区 ID、度中心性等），建议优先使用 GEXF。*

---

## 🚀 导入与初步设置

### 1. 导入数据
1. 打开 Gephi，在欢迎界面点击 **"Open Graph File..."**。
2. 选择你的 `.gexf` 文件。
3. 在弹出的 **Import Report** 窗口中：
   - **Graph Type**: 选择 "Directed" (有向图)。
   - **Edges merge strategy**: 保持默认 (Sum)。
   - 点击 **OK**。

### 2. 检查数据
导入后，你应该能看到一团黑色的节点云。在 **Data Laboratory** (数据实验室) 标签页中，检查 Nodes 表格是否包含以下关键属性：
- `Id` / `Label`: 实体名称
- `community`: 社区编号 (如果转换脚本已包含)
- `degree`: 度数 (可选，可在 Gephi 中重新计算)

---

## 🎨 可视化美化步骤 (核心流程)

为了制作出适合学术论文发表的高质量图片，请严格按照以下步骤操作：

### 步骤 1: 统计与计算 (Statistics)
我们需要先计算图的属性，以便后续用于上色和调整大小。
1. 在右侧 **Statistics** 面板中：
2. 点击 **Network Diameter** 旁的 "Run" -> 获得 `Betweenness Centrality` (介数中心性)。
3. 点击 **Modularity** 旁的 "Run" -> 获得 `Modularity Class` (用于社区上色)。
   * *注：如果你的 GEXF 文件中已经自带了 `community` 字段，可跳过 Modularity 计算。*

### 步骤 2: 布局调整 (Layout) —— 关键步骤
让图谱结构展开，不再是一团乱麻。
1. 在左下角 **Layout** 面板中，选择 **ForceAtlas 2**。
2. **核心参数调整** (重要):
   - **Scaling (缩放)**: 设置为 **20.0 ~ 50.0** (数值越大，节点越疏散)。
   - **Gravity (重力)**: 保持 1.0 或稍大。
   - **Prevent Overlap (防止重叠)**: **务必勾选**。
   - **Dissuade Hubs**: 勾选 (把中心大节点推开，避免遮挡)。
3. 点击 **Run**。让它运行 1-2 分钟，直到图谱形状稳定。
4. 点击 **Stop**。

> **微调技巧**: 如果运行完还是很挤，可以运行 **Expansion** 算法几次；最后运行 **Noverlap** 算法消除最后的重叠。

### 步骤 3: 节点外观 (Appearance)
让关键节点显眼，不同社区区分开。
1. 在左上角 **Appearance** 面板中选择 **Nodes**。
2. **颜色 (Color)** 🎨:
   - 选择 **Partition** -> 选择 `Modularity Class` (或者导入的 `community` 字段)。
   - 点击 **Apply**。节点将根据社区自动着色。
3. **大小 (Size)** ⭕:
   - 选择 **Ranking** -> 选择 `Degree` (度) 或 `PageRank`。
   - 设置 Min size: **10**, Max size: **50** (根据节点数量适当调整)。
   - 点击 **Apply**。大Hub节点会变大，边缘节点变小。

### 步骤 4: 标签处理 (Labels)
1. 在底部工具栏点击 **"T"** (Show Node Labels) 图标开启标签。
2. 标签通常会太大或太密。
3. **调整大小**: 点击标签颜色/大小设置，选择 "Node Size" 模式 (标签随节点大小变化)。
4. **防止遮挡**: 在 Layout 面板运行 **Label Adjust** 算法，自动推开重叠的标签。

---

## 📷 预览与导出 (Preview & Export)

这是生成最终图片的阶段。不要直接截图，要使用 Preview 功能。

### 1. 预览设置 (Preview Settings)
切换到 **Preview** 标签页。
1. **Presets (预设)**: 选择 "Default Curved"。
2. **Nodes (节点)**:
   - Border Color: 选择 **Parent** (跟随节点颜色) 或 **White** (白色描边更显精致)。
   - Opacity: 100。
3. **Edges (边)**:
   - **Opacity (透明度)**: 调低至 **20-30**。这是避免图谱看起来像“毛线球”的关键。
   - Color: Source / Target / Mixed (推荐 Mixed)。
   - Curved: 勾选 (曲线比直线更有美感)。
4. **Labels (标签)**:
   - Show Labels: 勾选。
   - Font: Arial 或 Times New Roman (保持与论文一致)。
5. 点击下方的 **Refresh** 按钮查看效果。

### 2. 导出图片
1. 点击左下角的 **Export** 按钮 (SVG/PDF/PNG)。
2. **推荐格式**:
   - **PDF/SVG**: 矢量图，适合插入 LaTeX/Word，无限放大不失真。
   - **PNG**: 如果必须用位图，请将分辨率 (Resolution) 设为 **3000 x 3000** 以上，以保证打印清晰度。

---

## ❓ 常见问题

**Q: 节点太多看不清怎么办？**
A: 使用右侧的 **Filters** (过滤器)。
- Topology -> Degree Range。
- 拖动滑块，过滤掉 Degree < 2 或 3 的边缘节点，只保留骨干网络。

**Q: 标签有些重叠怎么办？**
A: 导出 SVG 后，使用 Adobe Illustrator 或 Inkscape 手动微调重叠的标签位置。这是制作顶刊图片的必经之路。

---

## 相关文档

- [IMPORT_TO_NEO4J.md](IMPORT_TO_NEO4J.md) - 使用 Neo4j 进行查询和可视化
- [ARCHITECTURE.md](ARCHITECTURE.md) - 项目架构说明

---

## 更新日志

| 日期 | 更新内容 |
|------|---------|
| 2026-01-14 | 添加相关文档链接 |
