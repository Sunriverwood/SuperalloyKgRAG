# Copyright 2025 SUNRIVERWOOD
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json


def convert_to_neo4j_format(input_file, output_file):
    """
    将 NetworkX 风格的图 JSON 转换为 Neo4j 友好的导入格式。
    """

    # 辅助函数：处理 Neo4j 不支持的复杂嵌套类型
    def transform_value(value):
        if isinstance(value, (dict, list)):
            # 如果是纯基础类型的列表（如字符串列表），Neo4j 是支持的，保留原样
            if isinstance(value, list) and all(isinstance(x, (str, int, float, bool)) for x in value):
                return value
            # 如果是复杂对象（字典或包含字典的列表），转换为 JSON 字符串
            return json.dumps(value, ensure_ascii=False)
        return value

    try:
        # 1. 读取原始数据
        with open(input_file, 'r', encoding='utf-8') as f:
            source_data = json.load(f)

        neo4j_data = {
            "nodes": [],
            "relationships": []
        }

        # 2. 处理节点 (Nodes)
        if 'nodes' in source_data:
            for node in source_data['nodes']:
                # 提取核心字段
                node_id = node.get('id')
                node_type = node.get('type', 'Thing')  # 默认标签

                # 准备属性
                properties = {}

                # 处理 attributes 字段 (通常是嵌套字典，将其扁平化或合并)
                attributes = node.get('attributes', {})
                if isinstance(attributes, dict):
                    for k, v in attributes.items():
                        properties[k] = transform_value(v)

                # 处理其他字段
                exclude_keys = {'id', 'type', 'attributes'}
                for key, value in node.items():
                    if key not in exclude_keys:
                        properties[key] = transform_value(value)

                # 构建 Neo4j 节点对象
                neo4j_node = {
                    "id": node_id,
                    "labels": [node_type] if node_type else ["Thing"],
                    "properties": properties
                }
                neo4j_data["nodes"].append(neo4j_node)

        # 3. 处理关系 (Relationships)
        # 检查是用 'links' 还是 'edges' 作为键
        links_key = 'links' if 'links' in source_data else 'edges'

        if links_key in source_data:
            for link in source_data[links_key]:
                start_node = link.get('source')
                end_node = link.get('target')
                rel_type = link.get('relationship', 'RELATED_TO')  # 默认关系类型

                rel_props = {}
                exclude_keys = {'source', 'target', 'relationship'}

                for key, value in link.items():
                    if key not in exclude_keys:
                        rel_props[key] = transform_value(value)

                neo4j_rel = {
                    "start": start_node,
                    "end": end_node,
                    "type": rel_type,
                    "properties": rel_props
                }
                neo4j_data["relationships"].append(neo4j_rel)

        # 4. 写入结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(neo4j_data, f, ensure_ascii=False, indent=2)

        print(f"转换成功！生成节点: {len(neo4j_data['nodes'])}, 关系: {len(neo4j_data['relationships'])}")

    except Exception as e:
        print(f"转换过程中发生错误: {e}")


# 运行转换
convert_to_neo4j_format('../data/graphs/final_graph.json', '../data/graphs/neo4j_import_data.json')