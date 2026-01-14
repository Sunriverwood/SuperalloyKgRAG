# 实体合并机制说明文档

## 📋 目录

- [概述](#概述)
- [实体合并流程](#实体合并流程)
- [边的处理逻辑](#边的处理逻辑)
- [人工审核机制](#人工审核机制)
- [配置选项](#配置选项)

---

## 概述

实体合并（Entity Merge）是知识图谱构建中的关键步骤，用于识别和合并指向同一真实世界对象的不同实体节点。SuperalloyKgRAG 采用**嵌入相似度 + LLM 判断**的两阶段策略。

### 核心特性

- ✅ **高效相似度计算**: 支持 HNSW 和 Brute Force 算法
- ✅ **LLM 智能判断**: 避免误合并
- ✅ **边自动聚合**: 合并实体时自动处理关联边
- ✅ **人工审核**: 可选的抽样审核机制
- ✅ **完整溯源**: 保留合并历史和来源信息

---

## 实体合并流程

### 第一阶段：候选簇发现（基于嵌入相似度）

**代码位置**: `core/pipeline_qwen/graph_builder_qwen.py::build_candidate_clusters()`

#### 1. 计算实体嵌入

```python
# 为每个实体生成嵌入向量
embeddings = get_entity_embeddings(graph, entities)
```

#### 2. 相似度搜索

系统支持两种算法：

##### **HNSW (Hierarchical Navigable Small World)**
- **适用场景**: 大规模图谱（>5000 实体）
- **优势**: 快速近似搜索，时间复杂度 O(log N)
- **参数**:
  - `M`: 16（每层最大连接数）
  - `ef_construction`: 200（构建时搜索深度）
  - `ef_search`: 100（查询时搜索深度，最小值保证召回率）

##### **Brute Force**
- **适用场景**: 小规模图谱（<5000 实体）
- **优势**: 精确搜索，100% 召回率
- **时间复杂度**: O(N²)

#### 3. 候选簇生成

```python
# 找到每个实体的 top-k 最相似邻居（相似度 > min_sim）
topk = 10  # 默认值
min_sim = 0.9  # 默认相似度阈值

# 构建互为近邻的候选簇
clusters = build_mutual_neighbor_clusters(similarity_results)
```

**互为近邻逻辑**：
- 实体 A 和 B 互为 top-k 邻居 → 候选合并
- 使用并查集算法聚类

---

### 第二阶段：LLM 判断

**代码位置**: `core/pipeline_qwen/graph_builder_qwen.py::create_entity_merge_requests()`

#### Prompt 设计

每个候选簇提交给 LLM，包含：

```markdown
候选实体组:
1. Ni-base superalloy (Type: Material, Description: ...)
2. Nickel-based superalloy (Type: Material, Description: ...)
3. Ni基高温合金 (Type: Material, Description: ...)

任务: 判断哪些实体指向同一对象，给出合并分组和规范名称。
```

#### LLM 返回格式

```json
{
  "groups": [
    {
      "entities": ["Ni-base superalloy", "Nickel-based superalloy", "Ni基高温合金"],
      "canonical_name": "Nickel-based Superalloy",
      "reason": "这些都是镍基高温合金的不同表述"
    }
  ]
}
```

---

### 第三阶段：应用合并

**代码位置**: `core/pipeline_qwen/graph_builder_qwen.py::apply_entity_merge()`

#### 1. 构建映射关系

```python
alias2canon = {
    "Ni-base superalloy": "canonical_id_1",
    "Nickel-based superalloy": "canonical_id_1",
    "Ni基高温合金": "canonical_id_1"
}

canon_name_map = {
    "canonical_id_1": "Nickel-based Superalloy"
}
```

#### 2. 合并节点属性

```python
# 合并逻辑
merged_node = {
    "name": canonical_name,  # LLM 给出的规范名称
    "aliases": [所有别名],     # 收集所有别名
    "description": max(descriptions, key=len),  # 选择最长描述
    "type": most_common(types),  # 多数投票选择类型
    "provenance": [...],  # 合并溯源信息
    "is_disambiguated": True
}
```

---

## 边的处理逻辑

### 问题：合并实体后，原本的边怎么办？

**回答**: 系统会自动聚合边，相同的边（相同起点、终点、关系类型）会合并为一条边。

### 边聚合策略

**代码位置**: `core/pipeline_qwen/graph_builder_qwen.py::apply_entity_merge()` Line 1402-1430

#### 1. 重定向边的端点

```python
# 原图: A → B, A' → B'（其中 A' 是 A 的同义词，B' 是 B 的同义词）
# 合并后: A_canonical → B_canonical
```

#### 2. 边聚合规则

当多条边合并为一条时，系统支持两种聚合方式：

##### **Max 聚合（默认）**

```python
edge_agg = 'max'

# 例子：
# 边1: A → B, weight=0.8, description="通过固溶强化提高强度"
# 边2: A → B, weight=0.9, description="固溶强化机制"
# 边3: A → B, weight=0.7, description="固溶强化"

# 合并后:
# A → B, weight=0.9 (取最大值), description="通过固溶强化提高强度" (取最长)
```

##### **Sum 聚合**

```python
edge_agg = 'sum'

# 合并后:
# A → B, weight=2.4 (求和), description="通过固溶强化提高强度"
```

#### 3. 自环边过滤

```python
# 如果 A 和 B 合并为 C，原边 A → B 会变成 C → C
# 系统自动过滤这种自环边
if uu == vv: continue
```

#### 4. 边的唯一性键

```python
# 边通过三元组 (source, target, relationship) 唯一标识
key = (canonical_source, canonical_target, relationship_type)

# 相同键的边会合并
tmp_edges[key] = aggregated_edge
```

### 示例

**合并前**:
```
节点: IN718, Inconel-718, IN 718
边:
  IN718 --[contains]--> Ni (weight=0.8)
  Inconel-718 --[contains]--> Ni (weight=0.9)
  IN 718 --[contains]--> Cr (weight=0.7)
```

**合并后**:
```
节点: Inconel 718 (canonical)
  aliases: ["IN718", "Inconel-718", "IN 718"]
边:
  Inconel 718 --[contains]--> Ni (weight=0.9, 取max)
  Inconel 718 --[contains]--> Cr (weight=0.7)
```

---

## 人工审核机制

### 是否有人工审核开关？

**回答**: 是的，系统提供可选的人工审核功能。

### 配置方式

在 `config/settings.yaml` 中：

```yaml
graph_builder:
  # 人工审核配置
  enable_manual_review: False  # true 启用，false 禁用
  manual_review_sample_size: 5  # 抽样数量
  manual_review_output_dir: "data/reports/manual_review"  # 输出目录
```

### 审核流程

**代码位置**: `core/pipeline_qwen/graph_builder_qwen.py::run_entity_merge_stage()` Line 1895-1918

```python
if enable_manual_review:
    # 1. 从候选簇中随机抽样
    sample_clusters = random.sample(clusters, k=sample_size)
    
    # 2. 对比 LLM 判断结果
    review_report = run_entity_merge_review(
        graph=graph,
        clusters=sample_clusters,
        llm_groups=llm_merge_groups,
        sample_size=sample_size,
        output_dir=review_output_dir
    )
    
    # 3. 生成审核报告
    # - 对比候选簇和 LLM 分组
    # - 标记可疑合并
    # - 生成可视化对比图
```

### 审核报告内容

生成的报告包括：

1. **`entity_merge_review_report.json`**: 详细审核结果
   ```json
   {
     "sample_id": 1,
     "candidate_cluster": ["A", "B", "C"],
     "llm_decision": {
       "group": ["A", "B"],
       "canonical": "A_canonical"
     },
     "flags": ["C was excluded by LLM"]
   }
   ```

2. **`entity_merge_review_comparison.png`**: 可视化对比图

### 使用建议

- ✅ **首次运行**: 建议启用审核，抽样 5-10 个簇
- ✅ **评估质量**: 检查 LLM 的合并决策是否合理
- ✅ **调整阈值**: 根据审核结果调整 `entity_merge_min_sim`
- ❌ **生产环境**: 审核通过后可禁用以提高效率

---

## 配置选项

### 完整配置示例

```yaml
graph_builder:
  # 是否启用实体合并
  enable_entity_merge: True
  
  # 相似度搜索配置
  similarity_algorithm: "auto"  # auto | hnsw | brute_force
  auto_algorithm_threshold: 5000  # 自动切换阈值
  
  # 候选簇发现参数
  entity_merge_topk: 10  # 每个实体保留的最相似邻居数
  entity_merge_min_sim: 0.9  # 相似度阈值（0-1）
  
  # 批处理配置
  embedding_batch_size: 10000  # 每批次实体数量
  
  # HNSW 参数（可选）
  hnsw_params:
    auto_tune: true  # 自动调参
    quality_level: "balanced"  # fast | balanced | accurate
    M: 16
    ef_construction: 200
    ef_search: 100
  
  # 人工审核配置
  enable_manual_review: False
  manual_review_sample_size: 5
  manual_review_output_dir: "data/reports/manual_review"
```

### 参数调优建议

| 参数 | 小图谱 (<1000) | 中图谱 (1000-10000) | 大图谱 (>10000) |
|------|---------------|-------------------|----------------|
| `similarity_algorithm` | brute_force | auto | hnsw |
| `entity_merge_topk` | 5-10 | 10-15 | 10-20 |
| `entity_merge_min_sim` | 0.85-0.90 | 0.90-0.92 | 0.92-0.95 |
| `embedding_batch_size` | 1000 | 5000 | 10000 |

---

## 常见问题

### Q1: 为什么相似度很高的实体没有被合并？

**可能原因**:
1. LLM 判断它们不是同一对象
2. 不在对方的 top-k 邻居中（增加 `entity_merge_topk`）
3. 相似度低于阈值（降低 `entity_merge_min_sim`）

### Q2: 如何查看合并历史？

```python
# 检查节点的 provenance 字段
node_data = graph.nodes["entity_id"]
print(node_data["provenance"])
# 输出: [{"merged_from": ["alias1", "alias2", ...]}, ...]
```

### Q3: 合并后如何恢复原始实体？

合并操作**不可逆**，但可以通过以下方式追溯：
1. 查看 `provenance` 字段
2. 查看 `aliases` 字段
3. 重新运行 indexing 流程

### Q4: 边权重为什么使用 max 而不是 sum？

**设计考量**:
- **Max**: 保留最强证据，避免权重膨胀
- **Sum**: 累积证据强度，但可能过度放大

可以通过修改代码切换：
```python
graph = apply_entity_merge(graph, alias2canon, canon_name_map, edge_agg='sum')
```

---

## 相关文档

- [HNSW_USAGE_GUIDE.md](HNSW_USAGE_GUIDE.md) - HNSW 优化配置指南
- [RUN_INDEXING_GUIDE.md](RUN_INDEXING_GUIDE.md) - 索引流水线指南
- `core/pipeline_qwen/graph_builder_qwen.py` - 实现代码
- `config/settings.yaml` - 配置文件

---

## 更新日志

| 日期 | 更新内容 |
|------|---------|
| 2026-01-14 | 更新相关文档链接 |
| 2024-12-24 | 初始版本 |
