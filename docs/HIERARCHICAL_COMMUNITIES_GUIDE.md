# 递归分层社区发现使用指南

## 概述

本项目已实现**递归分层社区发现（Recursive Hierarchical Community Detection）**功能，用于支持 GraphRAG 的宏观-微观多粒度检索。

---

## 功能特性

### 1. 分层社区结构
- **Level 0**: 全图顶层社区
- **Level 1+**: 递归细分的子社区
- 自动构建父子关系树

### 2. 层级ID命名
```
根社区: "0"
一级社区: "0_1", "0_2", ...
二级社区: "0_1_1", "0_1_2", ...
```
便于通过前缀判断层级关系

### 3. 节点属性
每个节点包含以下属性：
- `community`: 最底层（最细粒度）社区ID
- `community_levels`: 所有层级的社区归属，如：
  ```python
  {
      'level_0': '0_1',
      'level_1': '0_1_2',
      'level_2': '0_1_2_3'
  }
  ```

### 4. 投影机制

投影机制确保**每个逻辑层级都覆盖图中的所有节点**，满足"集体穷尽"原则。

#### 两种社区类型

| 类型 | 定义 | `is_projected` |
|------|------|----------------|
| **真实社区** | 由上一层级通过 Leiden 算法分割产生 | `False` |
| **投影社区** | 从较浅层级投影到较深层级的社区 | `True` |

#### 示例

```
Level 2: 0_0_0 (真实社区) → 继续分割 → Level 3: 0_0_0_0, 0_0_0_1
Level 2: 0_1_0 (真实社区) → 无法分割 → Level 3: 0_1_0 (投影社区)
```

---

## 配置说明

在 `config/settings.yaml` 中的 `graph_builder` 部分：

```yaml
graph_builder:
  # === 分层社区发现配置 ===
  # 是否使用分层社区发现（True=分层，False=传统扁平化）
  use_hierarchical_communities: true
  
  # 最大递归层级（0, 1, 2, ...）
  max_community_level: 10
  
  # 最小社区节点数，少于此数不再细分
  min_community_size: 10
```

### 配置参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_hierarchical_communities` | bool | `true` | 是否启用分层社区发现 |
| `max_community_level` | int | `10` | 最大递归深度（0为根层级） |
| `min_community_size` | int | `10` | 社区节点数少于此值时停止细分 |

> **注意**: `max_cluster_size` 参数已移除，社区分割完全由 Leiden 算法的模块度优化自动决定。

---

## 使用示例

### 1. 在主流程中使用（自动）

如果正确配置了 `settings.yaml`，运行 `graph_builder_qwen.py` 会自动使用分层社区发现：

```bash
python -m core.pipeline_qwen.graph_builder_qwen
```

### 2. 独立使用

```python
import networkx as nx
from utils.recursive_leiden import (
    recursive_leiden_community_detection,
    apply_hierarchical_communities_to_graph,
    communities_to_dataframe
)

# 创建或加载图
G = nx.karate_club_graph()

# 执行递归社区发现
communities_list, node_community_map = recursive_leiden_community_detection(
    G=G,
    max_level=3,
    min_community_size=3
)

# 应用到图
G = apply_hierarchical_communities_to_graph(G, node_community_map)

# 转换为DataFrame查看
df = communities_to_dataframe(communities_list)
print(df[["community_id", "level", "parent_id", "node_count", "is_projected"]])
```

---

## 输出结果

### 1. 图节点属性

每个节点会被标注：
- `community`: 字符串，最细粒度社区ID
- `community_levels`: 字典，各层级社区ID
- `degree`: 整数，节点度数

### 2. 社区信息列表

返回的 `communities_list` 是一个列表，每个元素包含：

```python
{
    "community_id": "0_1_2",          # 社区ID
    "level": 2,                        # 层级
    "title": "Community 0_1_2",        # 标题（占位符）
    "parent_id": "0_1",                # 父社区ID
    "children_ids": ["0_1_2_1", ...], # 子社区ID列表
    "node_ids": ["node1", "node2", ...], # 包含的节点列表
    "is_projected": false              # 是否为投影社区
}
```

---

## 层级查询指南

### 层级选择策略

| 层级 | 语义 | Top-K | 适用场景 |
|------|------|-------|----------|
| Level 0 | 全局主题总览 | 2 | "超合金研究的整体发展趋势" |
| Level 1 | 主要研究方向 | 20 | "镍基高温合金的力学性能研究" |
| Level 2 | 具体子主题 | 50 | "单晶合金中γ'相的析出机制" |
| Level 3 | 细粒度概念 | 200 | "IN718在750°C下的γ''相演化" |

### 自动层级判定

系统支持根据查询内容自动选择最佳层级：

```bash
# 自动判定层级
python core/query_qwen/global_query_qwen.py "Your question here"

# 手动指定层级
python core/query_qwen/global_query_qwen.py --level 0 "Your question here"
```

---

## 递归终止条件

递归会在以下任一情况下停止：
1. 达到 `max_level`
2. 社区节点数 < `min_community_size`
3. Leiden 算法无法进一步细分（返回单一社区）

---

## 常见问题

### Q1: 如何只获取某一层级的社区？

```python
level_1_communities = [
    comm for comm in communities_list 
    if comm['level'] == 1
]
```

### Q2: 如何获取某个社区的所有子孙社区？

```python
def get_all_descendants(communities_list, parent_id):
    descendants = []
    for comm in communities_list:
        if comm['parent_id'] == parent_id:
            descendants.append(comm)
            descendants.extend(
                get_all_descendants(communities_list, comm['community_id'])
            )
    return descendants
```

### Q3: 如何控制递归深度？

调整 `max_community_level` 参数。Level 0 是根层级，所以 `max_community_level=2` 表示有3层（0, 1, 2）。

### Q4: 某些节点没有细分怎么办？

这是正常的。如果 Leiden 算法无法进一步细分，该社区就是叶子节点，会在更深层级作为投影社区出现。

---

## 相关文件

- 核心实现: `utils/recursive_leiden.py`
- 调用入口: `core/pipeline_qwen/graph_builder_qwen.py`
- 配置文件: `config/settings.yaml`
- 社区报告生成: 参见 [COMMUNITY_REPORT_GENERATION.md](COMMUNITY_REPORT_GENERATION.md)

---

## 更新日志

| 日期 | 更新内容 |
|------|---------|
| 2026-01-14 | 合并投影机制、层级查询文档；移除 `max_cluster_size` 参数说明 |
| 2025-12-24 | 初始版本，实现递归分层社区发现功能 |
