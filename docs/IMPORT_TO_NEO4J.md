# Neo4j 导入指南

## 概述

本文档说明如何将 SuperalloyKgRAG 生成的知识图谱导入 Neo4j 进行可视化和查询。

---

## 前置准备

1. **安装 Neo4j**: 下载并安装 [Neo4j Desktop](https://neo4j.com/download/) 或 Neo4j Community Edition
2. **安装 APOC 插件**: 确保已安装并启用 APOC 插件（用于加载 JSON 文件）
3. **准备数据**: 将 `neo4j_import_data.json` 文件放置在 Neo4j 的 `import` 目录中

---

## 启动 Neo4j

```bash
# 进入 Neo4j 安装目录的 bin 目录
cd neo4j/bin

# 启动 Neo4j 控制台
./neo4j console

# 浏览器访问: http://localhost:7474
# 用户名和密码参见 config/settings.yaml
```

---

## 导入步骤

### 步骤 1: 创建索引

```cypher
CALL apoc.load.json("file:///neo4j_import_data.json") YIELD value
UNWIND value.nodes AS node_data
WITH collect(DISTINCT node_data.label) AS unique_labels
WITH apoc.map.fromPairs([label IN unique_labels | [label, ['id']]]) AS schemaMap
CALL apoc.schema.assert(schemaMap, {}) YIELD label, key, unique
RETURN label, key, unique;
```

### 步骤 2: 导入节点

```cypher
CALL apoc.load.json("file:///neo4j_import_data.json") YIELD value
UNWIND value.nodes AS node_data
MERGE (n {id: node_data.id})
SET n += node_data.properties
WITH n, node_data
CALL apoc.create.addLabels(n, [node_data.label]) YIELD node
RETURN count(node) as nodes_created;
```

### 步骤 3: 创建关系

```cypher
CALL apoc.load.json("file:///neo4j_import_data.json") YIELD value
UNWIND value.relationships AS rel_data
MATCH (source {id: rel_data.start})
MATCH (target {id: rel_data.end})
CALL apoc.create.relationship(source, rel_data.type, rel_data.properties, target) YIELD rel
RETURN count(rel) as relationships_created;
```

---

## 清空数据库

如需重新导入，先清空所有节点和关系：

```cypher
MATCH (n)
DETACH DELETE n
```

---

## 常用查询

### 查看所有节点类型

```cypher
CALL db.labels() YIELD label
RETURN label
```

### 查看所有关系类型

```cypher
CALL db.relationshipTypes() YIELD relationshipType
RETURN relationshipType
```

### 查看图谱统计

```cypher
MATCH (n) RETURN count(n) as node_count;
MATCH ()-[r]->() RETURN count(r) as relationship_count;
```

---

## 相关文档

- [IMPORT_TO_GEPHI.md](IMPORT_TO_GEPHI.md) - 使用 Gephi 进行可视化
- [ARCHITECTURE.md](ARCHITECTURE.md) - 项目架构说明

---

## 更新日志

| 日期 | 更新内容 |
|------|---------|
| 2026-01-14 | 文档格式优化，添加常用查询 |
