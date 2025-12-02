# 允许单节点路径显示（恢复原始行为）

## 变更说明

已将代码恢复为允许显示单节点路径（即起点节点本身就是终点节点的情况）。

## 原因

之前的修复跳过了单节点路径，期望找到多跳推理链。但在实际图谱中：

1. **图谱连通性问题**：相关实体之间可能没有直接的边连接
2. **图谱稀疏性**：实体和关系抽取可能不完整
3. **查询语义**：有时查询的直接答案就是单个实体，不需要推理链

因此，显示单节点路径实际上是有意义的：
- 表示"查询直接匹配到这些相关实体"
- 至少能看到系统找到了哪些相关节点
- 帮助诊断图谱质量问题

## 代码变更

### 恢复的行为

```python
# 现在的行为
if start in end_set:
    paths.append([start])  # ✅ 添加单节点路径
    continue
```

### 之前的行为

```python
# 之前跳过单节点
if start in end_set:
    single_node_matches += 1
    logging.debug(f"Start node {start} is also in end_nodes, skipping")
    continue  # ❌ 跳过
```

## 如何解读结果

### 情况1：只有单节点路径

**输出示例**：
```
Reasoning Paths:
--------------------------------------------------------------------------------

Path 1 (confidence: 1.0000):
Single-node path: Nickel-based superalloys (no edges)

Path 2 (confidence: 1.0000):
Single-node path: Turbine blades (no edges)
```

**含义**：
- 系统找到了查询相关的实体
- 但这些实体之间**没有直接连接的边**
- 说明图谱中缺少"nickel → turbine blade"这样的关系

**可能的原因**：
1. 实体抽取正确，但关系抽取遗漏了
2. 原始文档中没有明确描述这种关系
3. 关系被抽取但在消歧/合并时丢失了

**解决方案**：
- 检查原始PDF中是否有相关描述
- 重新运行关系抽取，调整prompt
- 检查实体合并是否正确

### 情况2：混合路径（单节点+多跳）

**输出示例**：
```
Path 1 (confidence: 0.8542):
Path (score: 0.8542):
  Nickel --[is_component_of]--> Nickel-based superalloys
    (importance: 0.9100, score: 0.9200)
  Nickel-based superalloys --[used_in]--> Turbine blades
    (importance: 0.8800, score: 0.9000)
  → Turbine blades

Path 2 (confidence: 1.0000):
Single-node path: High-temperature alloys (no edges)
```

**含义**：
- 找到了一些多跳推理路径 ✅
- 同时也有直接匹配的相关实体
- 这是比较理想的情况

### 情况3：所有路径都是多跳

**输出示例**：
```
Path 1 (confidence: 0.8542):
Path (score: 0.8542):
  Nickel --[is_component_of]--> Nickel-based superalloys
    (importance: 0.9100, score: 0.9200)
  → Nickel-based superalloys

Path 2 (confidence: 0.7821):
Path (score: 0.7821):
  Superalloys --[application]--> Aerospace industry
    (importance: 0.8500, score: 0.8700)
  Aerospace industry --[uses]--> Turbine blades
    (importance: 0.8800, score: 0.9000)
  → Turbine blades
```

**含义**：
- 图谱质量良好，有丰富的连接 ✅
- 找到了完整的推理链
- 这是最理想的情况

## 诊断图谱质量

### 快速检查脚本

```python
import json
import networkx as nx

# 加载图谱
with open('data/graphs/final_graph.json', 'r', encoding='utf-8') as f:
    graph_data = json.load(f)
G = nx.node_link_graph(graph_data, directed=True)

print(f"图谱统计:")
print(f"  节点数: {G.number_of_nodes()}")
print(f"  边数: {G.number_of_edges()}")
print(f"  平均出度: {G.number_of_edges() / G.number_of_nodes():.2f}")

# 检查孤立节点
isolated = list(nx.isolates(G))
print(f"  孤立节点数: {len(isolated)}")

# 检查连通性
if G.is_directed():
    weakly_connected = nx.number_weakly_connected_components(G)
    print(f"  弱连通分量: {weakly_connected}")
else:
    connected = nx.number_connected_components(G)
    print(f"  连通分量: {connected}")

# 检查特定实体
def find_nodes_by_keyword(keyword):
    return [n for n, d in G.nodes(data=True) 
            if keyword.lower() in d.get('name', '').lower()]

nickel_nodes = find_nodes_by_keyword('nickel')
blade_nodes = find_nodes_by_keyword('blade')

print(f"\n实体检查:")
print(f"  包含'nickel'的节点: {len(nickel_nodes)}")
print(f"  包含'blade'的节点: {len(blade_nodes)}")

# 检查是否有路径连接
if nickel_nodes and blade_nodes:
    has_path = False
    for n in nickel_nodes:
        for b in blade_nodes:
            if nx.has_path(G, n, b):
                path = nx.shortest_path(G, n, b)
                print(f"\n找到路径: {len(path)} 跳")
                print(f"  起点: {G.nodes[n].get('name', n)}")
                print(f"  终点: {G.nodes[b].get('name', b)}")
                has_path = True
                break
        if has_path:
            break
    
    if not has_path:
        print("\n⚠️ 这些实体之间没有连接路径!")
```

### 改进图谱质量的建议

**如果发现连接稀疏**：

1. **调整关系抽取Prompt** (`config/prompts/text_to_graph.md`)
   - 增加更多关系类型示例
   - 明确要求抽取间接关系
   - 提高召回率（可能会降低精确率）

2. **降低实体合并阈值** (`config/settings.yaml`)
   ```yaml
   graph_builder:
     entity_merge_min_sim: 0.75  # 从0.82降低，允许更多合并
   ```

3. **增加chunk重叠** (`config/settings.yaml`)
   ```yaml
   loader:
     chunk_overlap: 200  # 从100增加到200，保留更多上下文
   ```

4. **检查原始文档**
   - 确认PDF解析正确
   - 检查是否有图表未被OCR
   - 验证关键段落是否被分块

## 当前状态

✅ 代码已恢复为允许单节点路径
✅ 诊断日志已更新
✅ 可以正常显示查询结果

## 使用方法

```bash
# 重新运行查询
python core/reasoning/run_reasoning_query.py \
    --query "What is the relationship between nickel and turbine blades?"
```

**期望输出**：
- 如果有多跳路径：显示完整的推理链
- 如果只有单节点：至少能看到相关实体
- 如果完全没有：说明查询匹配失败或图谱问题严重

## 总结

这个变更使系统更加实用：
- ✅ 总能显示一些结果（即使图谱不完美）
- ✅ 帮助诊断图谱质量
- ✅ 对用户更友好（不是完全空白）

缺点是失去了"必须有推理链"的严格性，但在当前图谱质量下这是更实用的选择。

