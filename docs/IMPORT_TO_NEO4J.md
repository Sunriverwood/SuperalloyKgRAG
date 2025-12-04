// 启动Neo4j并确保APOC插件已安装和启用
// 将json文件放置在Neo4j导入目录中（通常是neo4j/import/）
// 首先进入neo4j安装目录的bin目录，然后cmd运行./neo4j console，浏览器访问localhost:7474，用户名和密码参见settings.yaml

// 以下Cypher代码段演示如何使用APOC加载json文件并导入数据
// 单个json文件创建索引
CALL apoc.load.json("file:///neo4j_import_data.json") YIELD value
UNWIND value.nodes AS node_data
WITH collect(DISTINCT node_data.label) AS unique_labels
WITH apoc.map.fromPairs([label IN unique_labels | [label, ['id']]]) AS schemaMap
CALL apoc.schema.assert(schemaMap, {}) YIELD label, key, unique
RETURN label, key, unique;

// 单个json文件导入节点
CALL apoc.load.json("file:///neo4j_import_data.json") YIELD value
UNWIND value.nodes AS node_data
MERGE (n {id: node_data.id})
SET n += node_data.properties
WITH n, node_data
CALL apoc.create.addLabels(n, [node_data.label]) YIELD node
RETURN count(node) as nodes_created;

// 单个json文件创建关系
CALL apoc.load.json("file:///neo4j_import_data.json") YIELD value
UNWIND value.relationships AS rel_data
MATCH (source {id: rel_data.start})
MATCH (target {id: rel_data.end})
CALL apoc.create.relationship(source, rel_data.type, rel_data.properties, target) YIELD rel
RETURN count(rel) as relationships_created;

// 简单清空节点和关系
MATCH (n)
DETACH DELETE n