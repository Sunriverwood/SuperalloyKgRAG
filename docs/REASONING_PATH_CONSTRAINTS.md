# 图推理路径约束说明

> 更新时间：2026-04-23
>
> 主题：项目如何硬性限制推理不走虚假路径

## 概述

SuperalloyKgRAG 在图推理阶段对“虚假路径”有明确的结构性约束。这里的“虚假路径”特指：

- 路径中包含**知识图谱中不存在的边**
- 推理过程中通过模型传播、注意力或排序，隐式引入了**图结构外的连接**

当前实现的核心原则是：

> **推理只能沿知识图谱中的真实边传播、搜索和打分。**

因此，系统可以对“图里不存在的假边/假路径”形成硬约束；但对于“图里真实存在、但语义上牵强或和问题不够相关的路径”，只能通过评分和过滤机制抑制，不能绝对消除。

---

## 一、硬约束机制

### 1. 邻接掩码限制 GNN 注意力

在图数据加载阶段，系统会先构造二值邻接掩码：

- 代码位置：`core/reasoning/data_loader.py`
- 关键函数：`create_adjacency_mask()`
- 关键逻辑：
  - `mask[i, j] = 1` 表示图中存在边 `(i, j)`
  - `mask[i, j] = 0` 表示图中不存在边 `(i, j)`

实现证据：

- `core/reasoning/data_loader.py:221` 定义 `create_adjacency_mask`
- `core/reasoning/data_loader.py:229` 执行 `mask[edge_index[0], edge_index[1]] = 1.0`

在 RGAT 推理时，这个掩码会直接作用到注意力 logits：

- 代码位置：`core/reasoning/models/rgat.py`
- 关键逻辑：
  - 先取 `edge_mask = adjacency_mask[src_idx, dst_idx]`
  - 对不存在的边，将 attention logits 设为 `-1e9`

实现证据：

- `core/reasoning/models/rgat.py:183` 检查 `adjacency_mask`
- `core/reasoning/models/rgat.py:189` 执行 `torch.where(..., e, -1e9)`

这意味着：

- **不存在的边不会获得有效注意力权重**
- softmax 之后这类边的概率近似为 0
- GNN 不能借助注意力“发明”一条图外连接

---

### 2. PPR 转移矩阵只由真实边构建

PPR 扩散不是在全连接图上运行，而是直接用图中的真实边构建稀疏转移矩阵。

- 代码位置：`core/reasoning/inference/reasoner.py`
- 关键函数：`build_transition_matrix()`

实现证据：

- `core/reasoning/inference/reasoner.py:447` 定义 `build_transition_matrix`
- `core/reasoning/inference/reasoner.py:468` 用 `edge_index` 和 `edge_weights` 构造稀疏矩阵 `P`

由于 `edge_index` 本身只来自图中已有边，所以：

- 不存在的边不会进入转移矩阵
- 其对应转移概率天然为 0
- PPR 只能沿真实边做概率传播

---

### 3. 路径抽取只遍历图中的真实邻居

最终展示给用户的 reasoning path 不是由 LLM 自由生成，而是通过 BFS 在 `NetworkX` 图上显式搜索得到。

- 代码位置：
  - `core/reasoning/inference/reasoner.py`
  - `utils/graph_reasoning_utils.py`

实现证据：

- `core/reasoning/inference/reasoner.py:630` 定义 `extract_and_rank_paths`
- `core/reasoning/inference/reasoner.py:671` 调用 `extract_paths_bfs(...)`
- `utils/graph_reasoning_utils.py:204` 定义 `extract_paths_bfs`
- `utils/graph_reasoning_utils.py:252-260` 仅通过 `G.neighbors(current)` 扩展邻居

因此：

- BFS 只会沿 `NetworkX` 图中真实存在的边扩展
- 图中不存在的边不会进入候选路径
- `max_path_length` 还进一步限制了可搜索的最大跳数

---

### 4. 路径打分阶段再次验证边是否真实存在

即便某条异常路径混入候选集合，系统在路径打分时还会再次做一遍边存在性检查。

- 代码位置：`utils/graph_reasoning_utils.py`
- 关键函数：`score_path_by_importance()`

实现证据：

- `utils/graph_reasoning_utils.py:137` 定义 `score_path_by_importance`
- `utils/graph_reasoning_utils.py:164` 检查 `if not G.has_edge(u, v):`

该函数的处理方式是：

- 若路径中任意一段边不存在
- 立即返回整条路径得分 `0.0`

所以即使上游出现异常，最终输出阶段也会把非真实路径压成无效结果。

---

### 5. 最终路径还会经过最低分阈值过滤

在 `GraphReasoner.extract_and_rank_paths()` 中，路径排序完成后还会再做一轮最低分过滤：

- 代码位置：`core/reasoning/inference/reasoner.py`
- 关键参数：`min_path_score`

实现证据：

- `core/reasoning/inference/reasoner.py:715` 执行
  `ranked_paths = [p for p in ranked_paths if p.score >= self.min_path_score]`

这一步不是“图结构硬约束”，但它可以进一步去掉低质量、低置信度路径。

---

### 6. 答案生成阶段要求所有陈述可追溯

路径找出来之后，系统在交给 LLM 生成答案时，还通过 prompt 强制要求答案只能基于：

- 提供的实体
- 路径
- 关系
- 源文本

代码位置：`core/query_qwen/reasoning_query_qwen.py`

实现证据：

- `core/query_qwen/reasoning_query_qwen.py:605`
  `You MUST ONLY use information from the provided knowledge graph entities, their descriptions, relationships, and source texts`
- `core/query_qwen/reasoning_query_qwen.py:608`
  `Every statement in your answer must be traceable to the provided entities, paths, or source texts`

这一步不能代替图结构约束，但它能减少 LLM 在自然语言生成阶段脱离证据胡乱补充内容。

---

## 二、这套机制实际防住了什么

项目当前对以下问题有较强约束：

1. **模型自己发明了一条图里没有的边**
2. **PPR 在不存在的连接上扩散**
3. **BFS 搜出图外路径**
4. **最终答案引用了未提供的路径或来源**

换句话说，系统对“**结构上不存在的路径**”是有硬限制的。

---

## 三、这套机制防不住什么

项目**不能完全防住**以下情况：

1. 图中边确实存在，但本身抽取错误
2. 图中路径真实存在，但与当前问题语义相关性较弱
3. 路径每一跳都合法，但整体解释链条牵强
4. 查询起点/终点选得不准，导致找到“真实但误导”的路径

这些问题当前主要依靠以下机制缓解，而不是硬性禁止：

- `composite_importance`
- GNN attention
- `top_k_nodes`
- `max_path_length`
- `min_path_score`

因此，项目当前的能力边界可以概括为：

> **能硬性限制“图外假路径”，但不能绝对消除“图内弱相关路径”。**

---

## 四、结论

SuperalloyKgRAG 对“推理不走虚假路径”的保障，主要来自四层结构性约束：

1. 邻接掩码禁止 GNN 使用不存在的边
2. PPR 转移矩阵仅由真实边构建
3. BFS 仅在真实图结构上搜索路径
4. 路径打分阶段再次校验边存在性，不合法路径直接置零

在此基础上，再叠加：

- 最低路径分数过滤
- LLM 回答阶段的可追溯性约束

所以从当前代码实现看，系统对“**图里不存在的虚假路径**”有明确硬约束；但对“**图里存在但语义上不够合理的路径**”，仍然属于排序和过滤问题，而不是完全禁止问题。
