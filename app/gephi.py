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
import math
import networkx as nx

json_path = "../data/graphs/final_graph.json"  # 输入 JSON
gexf_path = "../data/graphs/final_graph.gexf"  # 输出 GEXF


def clean_value(v):
    """把属性值清洗成 GEXF 可接受的简单类型"""
    if v is None:
        return ""  # 可以改成 None 或者其他占位
    if isinstance(v, (str, bool, int)):
        return v
    if isinstance(v, float):
        # GEXF 不喜欢 NaN / inf，统一成 0 或者你想要的值
        if math.isnan(v) or math.isinf(v):
            return 0.0
        return v
    # list/dict/tuple/set/其他复杂对象 -> 直接转成字符串（这里用 JSON 方便人看）
    try:
        return json.dumps(v, ensure_ascii=False)
    except TypeError:
        return str(v)


# 1. 读取 JSON
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

directed = data.get("directed", False)
multigraph = data.get("multigraph", False)

# 2. 根据元数据创建合适的图类型
if directed:
    G = nx.MultiDiGraph() if multigraph else nx.DiGraph()
else:
    G = nx.MultiGraph() if multigraph else nx.Graph()

# 3. 加载节点（并清洗属性）
for n in data["nodes"]:
    node_id = n.get("id") or n.get("name")
    if node_id is None:
        continue

    attrs = {}
    for k, v in n.items():
        if k == "id":
            continue
        attrs[k] = clean_value(v)

    # 给 Gephi 一个 label（显示用）
    if "name" in n:
        attrs["label"] = n["name"]

    G.add_node(node_id, **attrs)

# 4. 加载边（并清洗属性）
for e in data["links"]:
    src = e["source"]
    tgt = e["target"]

    attrs = {}
    for k, v in e.items():
        if k in ("source", "target"):
            continue
        attrs[k] = clean_value(v)

    G.add_edge(src, tgt, **attrs)

# 5. 写 GEXF
nx.write_gexf(G, gexf_path, encoding="utf-8")
print(f"GEXF 文件已保存到: {gexf_path}")
