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

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any
import networkx as nx


def _to_simple_graph_if_needed(G: nx.Graph) -> nx.Graph:
    """
    若为 Multi(Graph/DiGraph)，将平行边合并为单边：仅保留 weight 最大的那条边及其完整属性。
    - 没有 weight 的边按 1.0 处理
    - 发生并列时，保留遇到的第一条（如需别的策略可改）
    """
    if not isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        return G  # 简单图直接返回
    H = nx.DiGraph() if G.is_directed() else nx.Graph()
    # 记录每对(u,v)当前选择的最佳边属性
    best = {}
    # MultiGraph/MultiDiGraph: 遍历所有平行边
    for u, v, k, data in G.edges(keys=True, data=True):
        try:
            w = float(data.get("weight", 1.0))
        except (TypeError, ValueError):
            w = 1.0
        key_uv = (u, v)  # 对于 DiGraph，方向敏感；对于 Graph，NetworkX 自己会处理无序对
        if key_uv not in best or w > best[key_uv][0]:
            # 深拷贝以免后续修改影响原图
            best[key_uv] = (w, dict(data))
    # 将挑选出的最大权重边写入 H
    for (u, v), (w, attr) in best.items():
        # 确保 weight 与选中值一致（即使原 data 没有 weight 也写入）
        attr = dict(attr)
        attr["weight"] = w
        H.add_edge(u, v, **attr)
    return H


def _calculate_importance_for_community_worker(subgraph: nx.Graph) -> Dict[tuple, float]:
    """
    (CPU Worker) 计算单个社区子图中边的介数中心性。
    """
    if subgraph.number_of_edges() == 0:
        return {}

    # MultiGraph 转简单图，避免 not implemented 错误
    subgraph = _to_simple_graph_if_needed(subgraph)

    # 使用 networkx 计算边介数中心性（返回 {(u, v): score}）
    return nx.edge_betweenness_centrality(subgraph, normalized=True)


def _calculate_parallel_importance_cpu(graph: nx.Graph, communities: List[List[Any]]) -> Dict[tuple, float]:
    """
    (CPU Orchestrator) 使用多进程并行计算所有社区的边介数中心性。
    """
    cpu_count = os.cpu_count() or 1
    logging.info(f"启动 CPU 并行重要性计算，将使用最多 {cpu_count} 个核心处理 {len(communities)} 个社区...")
    all_centrality_scores: Dict[tuple, float] = {}

    with ProcessPoolExecutor(max_workers=cpu_count) as executor:
        futures = {
            # 关键：传 **副本**，不要把 subgraph 视图对象直接丢给子进程
            executor.submit(_calculate_importance_for_community_worker, graph.subgraph(c).copy()): tuple(c)
            for c in communities if len(c) > 1
        }

        for future in as_completed(futures):
            try:
                community_centrality = future.result()
                # 先合并到总表中
                all_centrality_scores.update(community_centrality)
            except Exception as e:
                logging.error(f"一个社区的重要性计算失败: {e}", exc_info=True)

    return all_centrality_scores


def calculate_community_importance(
    graph: nx.Graph,
    communities: List[List[Any]],
    weight_alpha: float = 0.6,
) -> nx.Graph:
    """
    计算社区内关系的重要性，并将其作为属性添加到图中。
    返回：带有 'betweenness_centrality' 和 'composite_importance' 的图。
    """
    # 1) 并行计算介数中心性
    centrality_scores = _calculate_parallel_importance_cpu(graph, communities)

    if not centrality_scores:
        logging.warning("未能计算出任何关系的重要性分数，跳过更新。")
        return graph

    # 2) 只对主图里真实存在的边进行更新，避免类型/不存在错误
    centrality_scores_valid = {}
    if isinstance(graph, (nx.MultiGraph, nx.MultiDiGraph)):
        # MultiGraph：允许用二元组 (u,v) 对所有 key 统一赋值
        for (u, v), s in centrality_scores.items():
            if graph.has_edge(u, v):
                centrality_scores_valid[(u, v)] = float(s)
    else:
        for (u, v), s in centrality_scores.items():
            if graph.has_edge(u, v) or graph.has_edge(v, u):
                # 无向图两端顺序都可；有向图按 (u,v) 检查
                key = (u, v) if graph.has_edge(u, v) else (v, u)
                centrality_scores_valid[key] = float(s)

    if not centrality_scores_valid:
        logging.warning("中心性结果与主图边集不匹配，未进行任何更新。")
        return graph

    nx.set_edge_attributes(graph, centrality_scores_valid, "betweenness_centrality")
    logging.info(f"已将 {len(centrality_scores_valid)} 条边的介数中心性分数更新到图中。")

    # 3) 计算综合重要性分数（对权重和介数中心性都做 min-max 归一化）
    # 收集已有权重
    weights = []
    for _, _, data in graph.edges(data=True):
        if "weight" in data:
            try:
                weights.append(float(data["weight"]))
            except (TypeError, ValueError):
                pass

    w_min = min(weights) if weights else 0.0
    w_max = max(weights) if weights else 1.0
    denom = (w_max - w_min) if (w_max > w_min) else 1.0

    # 收集介数中心性并做 min-max 归一化准备
    central_values = []
    for _, _, data in graph.edges(data=True):
        try:
            central_values.append(float(data.get("betweenness_centrality", 0.0)))
        except (TypeError, ValueError):
            central_values.append(0.0)

    c_min = min(central_values) if central_values else 0.0
    c_max = max(central_values) if central_values else 1.0
    c_denom = (c_max - c_min) if (c_max > c_min) else 1.0

    composite_scores = {}
    for u, v, data in graph.edges(data=True):
        raw_w = data.get("weight", 0.0)
        try:
            raw_w = float(raw_w)
        except (TypeError, ValueError):
            raw_w = 0.0

        # 对权重做 0..1 的 min-max 归一化并截断
        normalized_weight = (raw_w - w_min) / denom
        normalized_weight = max(0.0, min(1.0, normalized_weight))

        # 对介数中心性做 0..1 的 min-max 归一化并截断
        try:
            centrality = float(data.get("betweenness_centrality", 0.0))
        except (TypeError, ValueError):
            centrality = 0.0
        normalized_centrality = (centrality - c_min) / c_denom
        normalized_centrality = max(0.0, min(1.0, normalized_centrality))

        score = (weight_alpha * normalized_centrality) + ((1.0 - weight_alpha) * normalized_weight)
        composite_scores[(u, v)] = score

    nx.set_edge_attributes(graph, composite_scores, "composite_importance")
    logging.info("已计算并添加综合重要性分数 'composite_importance'。")

    return graph
