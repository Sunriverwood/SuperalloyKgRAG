# 图推理系统 - 完整指南

## 📋 概述

图推理系统是 SuperalloyKgRAG 的核心功能模块，实现基于知识图谱的多跳推理和路径发现。系统采用自监督学习方式，结合 Query-Aware RGAT 图神经网络和 Personalized PageRank 算法，实现可解释的推理查询。

> **⚠️ 重要提示**: 推理查询应通过 `router_qwen.py` 统一入口使用，路由器会自动判断是否需要推理模式。

## 🏗️ 系统架构

### 核心组件

```
┌─────────────────────────────────────────────────────────────────┐
│                     图推理系统架构                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │  数据加载模块     │  │  图神经网络模块   │  │  推理引擎    │  │
│  │  data_loader.py  │  │  models/rgat.py  │  │  reasoner.py │  │
│  └──────────────────┘  └──────────────────┘  └──────────────┘  │
│          ↓                      ↓                    ↓          │
│  • 加载 final_graph.json    • Query-Aware RGAT   • PPR 传播   │
│  • 提取向量 enriched.db     • 多头注意力机制     • BFS 路径搜索│
│  • 构建邻接矩阵            • 边权重整合         • 路径评分     │
│                                                                 │
│  ┌──────────────────┐                    ┌──────────────────┐  │
│  │  训练模块        │                    │  查询处理模块     │  │
│  │  trainer.py      │                    │  reasoning_      │  │
│  └──────────────────┘                    │  query_qwen.py   │  │
│          ↓                                └──────────────────┘  │
│  • 边重构（Link Prediction）                      ↓            │
│  • 对比学习（Contrastive）                  • 查询编码          │
│  • 伪查询任务（Pseudo Query）              • 节点评分          │
│                                            • LLM 答案生成      │
└─────────────────────────────────────────────────────────────────┘
```

### 数据流程

```
final_graph.json + enriched.db
         ↓
    数据加载器 (data_loader.py)
         ↓
    图数据结构 (NetworkX + PyTorch Geometric)
         ↓
   ┌─────────┴─────────┐
   ↓                   ↓
训练阶段            推理阶段
   ↓                   ↓
自监督训练        查询 → PPR/GNN → 路径 → 答案
   ↓
develop.pt
```

## 🔧 配置说明

在 `config/settings.yaml` 中配置推理系统参数：

```yaml
reasoning:
  model:
    hidden_dim: 256          # 隐藏层维度
    num_layers: 3            # RGAT 层数
    num_heads: 4             # 注意力头数
    dropout: 0.1             # Dropout 比例
    use_edge_weights: true   # 使用 composite_importance
  
  training:
    num_epochs: 100          # 训练轮数
    learning_rate: 0.001     # 学习率
    batch_size: 512          # 批次大小
    loss_weights:
      link_prediction: 1.0   # 边重构损失权重
      contrastive: 0.5       # 对比学习损失权重
      pseudo_query: 1.0      # 伪查询损失权重
  
  inference:
    top_k_nodes: 20          # Top-K 相关节点
    ppr_alpha: 0.15          # PPR 重启概率
    max_path_length: 3       # 最大路径长度
    max_paths_per_query: 10  # 每次查询最大路径数
```

## 🚀 使用指南

### 方式1: 通过路由器使用（推荐）

**统一查询入口**，自动判断是否需要推理模式：

```bash
# 交互模式
$env:PYTHONPATH="."; python core/query_qwen/router_qwen.py

# 命令行模式
$env:PYTHONPATH="."; python core/query_qwen/router_qwen.py --query "镍和涡轮叶片有什么关系？"
```

**路由器会自动识别推理类型查询**，例如：
- "X 和 Y 有什么关系？" → REASONING
- "为什么 X 会影响 Y？" → REASONING
- "X 如何连接到 Y？" → REASONING

### 方式2: 直接使用推理模块

> **注意**: 仅在调试或特殊需求时直接使用，日常查询请使用路由器。

#### 首次使用：训练模型

```bash
# 基础训练（50 轮）
python core/query_qwen/reasoning_query_qwen.py --train --epochs 50

# 自定义参数训练
python core/query_qwen/reasoning_query_qwen.py --train --epochs 100

# 强制重新训练（即使模型已存在）
python core/query_qwen/reasoning_query_qwen.py --train --force-train --epochs 100
```

训练完成后，模型保存路径由 `config/settings.yaml` 的 `reasoning.output.model_path` 控制（当前默认：`data/reasoning/develop.pt`）。

#### 执行推理查询

**交互模式**（推荐）：

```bash
# 直接运行，进入交互界面
python core/query_qwen/reasoning_query_qwen.py
```

交互流程示例：
```
================================================================================
Graph Reasoning Query System - Interactive Mode
================================================================================

✓ Found trained model: D:\...\data\reasoning\develop.pt

Enter your query (or 'quit' to exit): 镍和涡轮叶片有什么关系？

Reasoning method (ppr/gnn) [ppr]: ppr
Generate LLM answer? (yes/no) [yes]: yes
Save results to file? (yes/no) [no]: no

Processing query...
```

**命令行模式**：

```bash
# 基本查询
python core/query_qwen/reasoning_query_qwen.py --query "镍和涡轮叶片有什么关系？"

# 指定推理方法
python core/query_qwen/reasoning_query_qwen.py --query "..." --method gnn

# 仅获取推理路径，不生成 LLM 答案
python core/query_qwen/reasoning_query_qwen.py --query "..." --no-llm

# 保存结果到文件
python core/query_qwen/reasoning_query_qwen.py --query "..." --output results.json
```

## 🧠 工作原理

### 自监督训练

系统通过三种自监督任务学习图结构知识：

#### 1. 边重构（Link Prediction）

```
目标: 预测节点对之间是否存在边

正样本: 图中存在的边（权重 = composite_importance）
负样本: 随机采样的不存在的边

损失函数: BCE Loss，正样本带权重
```

#### 2. 图对比学习（Contrastive Learning）

```
目标: 学习鲁棒的节点表示

方法: 
1. 创建两个图增强视图（边删除、特征噪声）
2. 同一节点的两个视图应该相似
3. 不同节点应该不相似

损失函数: InfoNCE
```

#### 3. 伪查询任务（Pseudo Query Matching）

```
目标: 训练查询-实体匹配函数

对每个三元组 (头实体, 关系, 尾实体):
  查询 = encode(头实体描述 + 关系描述)
  正样本 = 尾实体
  负样本 = 随机实体

损失函数: Triplet Loss
```

### 推理流程

#### PPR 方法（Personalized PageRank）

```
1. 初始化节点分布:
   π_0[i] = similarity(query, node_i)

2. 迭代传播:
   π_{k+1} = α·π_0 + (1-α)·π_k·P
   
   其中:
   - α: 重启概率（默认 0.15）
   - P: 转移矩阵（基于 composite_importance 归一化）

3. 选取 Top-K 节点
4. BFS 搜索路径
5. 路径评分 = Π(composite_importance × attention)
```

**适用场景**: 连接性查询，如 "X 和 Y 有什么关系？"

#### GNN 方法（Query-Aware RGAT）

```
1. 查询编码:
   q = encode_query(query_text)

2. 图神经网络传播:
   h^(l+1) = RGAT(h^(l), edge_index, edge_type, edge_weights, q)
   
   注意力机制:
   α_{ij} = softmax(attention(h_i, h_j, r_{ij}, q) × w_{ij})
   
   其中 w_{ij} = composite_importance

3. 节点评分:
   score_i = Matcher(q, h_i^(L))

4. 选取 Top-K 节点
5. BFS 搜索路径
6. 路径评分
```

**适用场景**: 因果推理，如 "为什么 X 会影响 Y？"

### 图约束机制

系统严格遵守图结构约束，确保推理可靠：

```
✅ 注意力掩码: 不存在的边 → attention = -∞
✅ PPR 转移: P[i,j] = 0 如果边 (i,j) 不存在
✅ 路径搜索: BFS 仅沿 NetworkX 图中的真实边
✅ 路径评分: 包含不存在的边 → score = 0
```

## 📊 输出格式

### 屏幕输出

```
================================================================================
REASONING RESULTS
================================================================================

Query: 镍和涡轮叶片有什么关系？

Top Relevant Entities:
--------------------------------------------------------------------------------
 1. 镍基超合金                                         (score: 3.89)
 2. 涡轮叶片                                           (score: 3.65)
 3. Inconel 718                                       (score: 3.21)

Reasoning Paths:
--------------------------------------------------------------------------------

Path 1 (confidence: 0.89):
  镍 --[contains]--> Inconel 718 --[used_in]--> 涡轮叶片

Path 2 (confidence: 0.76):
  镍 --[provides]--> 高温强度 --[required_by]--> 涡轮叶片

Final Answer:
--------------------------------------------------------------------------------
镍是制造涡轮叶片的关键材料。主要关系包括：
1. 镍是 Inconel 718 等镍基超合金的主要成分...
2. 镍提供的高温强度是涡轮叶片的关键性能要求...
```

### JSON 输出（--output）

```json
{
  "query": "镍和涡轮叶片有什么关系？",
  "top_nodes": [
    {
      "id": "entity_123",
      "name": "镍基超合金",
      "score": 3.89
    }
  ],
  "paths": [
    {
      "path": ["镍", "Inconel 718", "涡轮叶片"],
      "score": 0.89,
      "edge_types": ["contains", "used_in"],
      "explanation": "镍 --[contains]--> Inconel 718 --[used_in]--> 涡轮叶片"
    }
  ],
  "num_paths": 2,
  "answer": "镍是制造涡轮叶片的关键材料..."
}
```

## 🛠️ 性能优化

### 训练优化

```bash
# GPU 加速（10-100倍速度提升）
python core/query_qwen/reasoning_query_qwen.py --train --device cuda

# 调整批次大小
# 大批次（1024-2048）: 更好的收敛
# 小批次（256-512）: 内存受限时
```

### 推理优化

| 参数 | 作用 | 建议值 |
|------|------|--------|
| `top_k_nodes` | 控制候选节点数 | 10-20 |
| `ppr_alpha` | 控制探索范围 | 0.1（全局）～ 0.3（局部）|
| `max_path_length` | 最大路径跳数 | 2-4 |
| `max_paths_per_query` | 返回路径数 | 5-10 |

## ❓ 常见问题

### Q1: 何时需要训练模型？

**首次使用或知识图谱更新后**需要训练：

```bash
# 检查模型是否存在 (PowerShell)
Get-Item data/reasoning/develop.pt

# 如果不存在，训练模型
python core/query_qwen/reasoning_query_qwen.py --train --epochs 50
```

### Q2: 训练需要多长时间？

取决于图规模和硬件：
- **小图（<10K节点）**: 5-15 分钟（CPU）
- **中图（10K-100K）**: 30-60 分钟（GPU）
- **大图（>100K）**: 2-4 小时（GPU）

### Q3: PPR 和 GNN 如何选择？

**由路由器自动选择**，或手动指定：
- **PPR**: 快速、适合连接性查询、不需要训练模型
- **GNN**: 慢速、适合因果推理、需要预先训练模型

### Q4: 推理查询没有返回路径？

检查以下原因：
1. 起点和终点节点在图中是否连通？
2. `max_path_length` 是否足够？
3. 邻接矩阵是否正确构建？

### Q5: 如何可视化推理路径？

可以导入 Neo4j 进行可视化：

```bash
# 1. 导入知识图谱到 Neo4j（见 Import_json_into_Neo4j.md）
# 2. 在 Neo4j Browser 中查询路径
MATCH path = (start {name: "镍"})-[*1..3]-(end {name: "涡轮叶片"})
RETURN path
```

## 📁 目录结构

```
core/reasoning/
├── __init__.py
├── data_loader.py          # 图数据加载
├── models/
│   ├── __init__.py
│   └── rgat.py            # Query-Aware RGAT 模型
├── training/
│   ├── __init__.py
│   └── trainer.py         # 自监督训练器
└── inference/
    ├── __init__.py
    └── reasoner.py        # 推理引擎（PPR + 路径搜索）

core/query_qwen/
└── reasoning_query_qwen.py # 统一查询入口

utils/
└── graph_reasoning_utils.py # 工具函数

data/reasoning/
└── develop.pt              # 默认训练好的模型（自动生成）
```

## 🔗 相关文档

- [ARCHITECTURE.md](ARCHITECTURE.md) - 查看推理查询在整体架构中的位置
- [RUN_INDEXING_GUIDE.md](RUN_INDEXING_GUIDE.md) - 了解如何构建知识图谱

---

**提示**: 日常使用推荐通过 `router_qwen.py` 统一入口，路由器会自动判断并选择最优查询模式。

