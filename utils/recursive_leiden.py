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

"""
递归分层社区发现模块
实现基于 Leiden 算法的递归分层社区检测，支持 GraphRAG 的宏观-微观多粒度检索。
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import networkx as nx
import pandas as pd
import igraph as ig
import leidenalg as la


class HierarchicalCommunity:
    """社区层级结构"""
    def __init__(self, community_id: str, level: int, parent_id: Optional[str] = None):
        self.community_id = community_id
        self.level = level
        self.parent_id = parent_id
        self.children_ids: List[str] = []
        self.node_ids: List[str] = []
        self.title = f"Community {community_id}"
        self.is_projected = False  # 标记是否为投影社区（叶子社区在更深层级的投影）

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "community_id": self.community_id,
            "level": self.level,
            "title": self.title,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids.copy(),
            "node_ids": self.node_ids.copy(),
            "is_projected": self.is_projected
        }


def _run_leiden_on_subgraph(subgraph: nx.Graph) -> List[List[int]]:
    """
    在子图上运行 Leiden 算法
    
    Args:
        subgraph: NetworkX 无向图
    
    Returns:
        社区列表，每个社区是节点索引的列表
    """
    if len(subgraph.nodes) == 0:
        return []
    
    try:
        # 转换为 igraph
        ig_graph = ig.Graph.from_networkx(subgraph)
        
        # 运行 Leiden 算法
        partition = la.find_partition(ig_graph, la.ModularityVertexPartition)
        
        # 返回社区列表
        return [list(community) for community in partition]
    except Exception as e:
        logging.warning(f"Leiden 算法执行失败: {e}，返回整个图作为一个社区")
        return [list(range(len(subgraph.nodes)))]


def recursive_leiden_community_detection(
    G: nx.Graph,
    max_level: int = 3,
    min_community_size: int = 10
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    递归分层社区发现 (GraphRAG标准：Level 0 = 根/最粗粒度，Level N = 叶子/最细粒度)

    采用相互排斥、集体穷尽（mutually exclusive, collectively exhaustive）的方式递归划分社区。
    每一层都是对完整数据集的独立划分，确保所有节点都被覆盖且不重复。

    层级定义：
    - Level 0: 根层级，整个图作为1个社区
    - Level 1: 对Level 0使用Leiden算法进行首次分割
    - Level 2+: 对每个社区继续使用Leiden算法递归分割，直到无法再分或达到终止条件
    - Level N: 叶子层级，最细粒度的社区（无法再分割的社区）

    终止条件：
    1. 达到最大层级 max_level
    2. 社区节点数小于 min_community_size
    3. Leiden算法无法进一步分割（返回单一社区）

    Args:
        G: NetworkX 图对象（无向图）
        max_level: 最大递归层级（防止无限递归）
        min_community_size: 最小社区节点数，少于此数则不再细分
    
    Returns:
        tuple: (communities_list, node_community_map)
            - communities_list: 社区信息列表，每个元素包含社区的所有信息
            - node_community_map: 节点到社区的映射，包含 'community' 和 'community_levels'
    """
    logging.info(f"开始递归分层社区发现 (max_level={max_level}, min_community_size={min_community_size})...")

    if len(G.nodes) == 0:
        logging.warning("图中没有节点，无法进行社区发现")
        return [], {}
    
    if len(G.edges) == 0:
        logging.warning("图中没有边，无法进行社区发现")
        return [], {}
    
    # 确保是无向图
    if isinstance(G, nx.DiGraph):
        G = G.to_undirected()
    
    # 存储所有社区
    all_communities: List[HierarchicalCommunity] = []
    # 节点列表
    node_list = list(G.nodes())

    # 节点到所属层级社区的映射 {node_name: {level: community_id}}
    node_level_communities: Dict[str, Dict[int, str]] = {node: {} for node in node_list}
    # 节点到最底层社区的映射 {node_name: community_id}
    node_final_community: Dict[str, str] = {}
    # 跟踪每个层级的最大社区数量（用于确保全图覆盖）
    max_level_reached = 0

    # Step 1: 创建Level 0 - 整个图作为单个根社区
    root_community = HierarchicalCommunity(
        community_id="0",
        level=0,
        parent_id=None
    )
    root_community.node_ids = node_list.copy()
    all_communities.append(root_community)

    # 所有节点都属于Level 0的根社区
    for node_name in node_list:
        node_level_communities[node_name][0] = "0"

    logging.info(f"Level 0: 创建根社区，包含 {len(node_list)} 个节点")

    # Step 2: 递归函数 - 从Level 1开始使用Leiden分割
    def _recursive_detect(
        subgraph: nx.Graph,
        parent_id: str,
        current_level: int,
        id_prefix: str
    ) -> None:
        """递归执行社区检测"""
        nonlocal max_level_reached

        # 更新最大层级
        max_level_reached = max(max_level_reached, current_level)

        # 检查终止条件
        if current_level > max_level:
            logging.debug(f"达到最大层级 {max_level}，停止递归")
            return
        
        if len(subgraph.nodes) < min_community_size:
            logging.debug(f"社区节点数 {len(subgraph.nodes)} 小于最小值 {min_community_size}，停止递归")
            return

        # 运行 Leiden 算法进行分割
        communities = _run_leiden_on_subgraph(subgraph)
        
        # 如果Leiden只返回1个社区（无法进一步分割），停止递归
        if len(communities) <= 1:
            logging.debug(f"Leiden无法进一步分割社区 {parent_id}，停止递归")
            return

        # Level 1 特殊处理：确保每个社区的大小都 >= min_community_size
        if current_level == 1:
            subgraph_node_list = list(subgraph.nodes())
            # 过滤出符合大小要求的社区和不符合的社区
            valid_communities = []
            small_communities = []

            for community_indices in communities:
                community_size = len(community_indices)
                if community_size >= min_community_size:
                    valid_communities.append(community_indices)
                else:
                    small_communities.append(community_indices)

            # 如果有过小的社区，将其合并到最大的有效社区中
            if small_communities and valid_communities:
                # 找到最大的有效社区
                largest_community_idx = max(range(len(valid_communities)),
                                          key=lambda i: len(valid_communities[i]))
                # 将所有小社区的节点合并到最大社区
                for small_comm in small_communities:
                    valid_communities[largest_community_idx].extend(small_comm)

                logging.info(f"Level 1: 合并了 {len(small_communities)} 个小社区（< {min_community_size} 节点）到最大社区")
                communities = valid_communities
            elif not valid_communities:
                # 如果没有符合要求的社区，说明分割失败，停止递归
                logging.debug(f"Level 1: 所有社区都小于 {min_community_size} 节点，停止递归")
                return

        # logging.info(f"Level {current_level}: 将父社区 {parent_id} ({len(subgraph.nodes)} 节点) 分割为 {len(communities)} 个子社区")

        # 处理每个社区
        for idx, community_node_indices in enumerate(communities):
            # 生成社区ID (使用父ID作为前缀)
            community_id = f"{parent_id}_{idx}"

            # 获取社区中的实际节点名称
            subgraph_node_list = list(subgraph.nodes())
            community_nodes = [subgraph_node_list[i] for i in community_node_indices]
            
            # 创建社区对象
            community = HierarchicalCommunity(
                community_id=community_id,
                level=current_level,
                parent_id=parent_id
            )
            community.node_ids = community_nodes
            
            # 添加到总列表
            all_communities.append(community)
            
            # 更新父社区的子社区列表
            for comm in all_communities:
                if comm.community_id == parent_id:
                    comm.children_ids.append(community_id)
                    break

            # 更新节点的层级社区映射
            for node_name in community_nodes:
                node_level_communities[node_name][current_level] = community_id
            
            # 检查是否需要继续递归（移除max_cluster_size限制）
            should_recurse = (
                current_level < max_level and
                len(community_nodes) >= min_community_size
            )
            
            if should_recurse:
                logging.debug(f"社区 {community_id} 有 {len(community_nodes)} 个节点，继续递归到 Level {current_level + 1}...")
                # 创建子图并递归
                community_subgraph = subgraph.subgraph(community_nodes).copy()
                _recursive_detect(
                    subgraph=community_subgraph,
                    parent_id=community_id,
                    current_level=current_level + 1,
                    id_prefix=community_id
                )
            else:
                # 这是叶子社区，标记为最终社区
                logging.debug(f"社区 {community_id} 是叶子社区 ({len(community_nodes)} 节点)")
                for node_name in community_nodes:
                    node_final_community[node_name] = community_id
    
    # Step 3: 从Level 0的根社区开始递归（从Level 1开始使用Leiden分割）
    _recursive_detect(
        subgraph=G,
        parent_id="0",
        current_level=1,  # 从Level 1开始应用Leiden
        id_prefix="0"
    )

    # Step 4: 投影机制 - 确保每个层级都覆盖所有节点（集体穷尽原则）
    # 对于在某层级就停止分割的叶子社区，需要将其投影到后续所有层级
    logging.info(f"开始投影叶子社区到所有层级（确保集体穷尽）...")

    # 用于跟踪已创建的投影社区，避免重复创建
    # key: (source_community_id, target_level), value: projected_community_id
    projected_communities_map: Dict[Tuple[str, int], str] = {}

    for level in range(1, max_level_reached + 1):
        # 检查每个节点在当前层级是否有归属
        for node_name in node_list:
            if level not in node_level_communities[node_name]:
                # 该节点在当前层级没有归属，需要投影其叶子社区
                # 找到该节点在更早层级的最深归属
                previous_levels = [lv for lv in node_level_communities[node_name].keys() if lv < level]
                if previous_levels:
                    # 使用最深的已知层级作为投影源
                    source_level = max(previous_levels)
                    source_community_id = node_level_communities[node_name][source_level]

                    # 检查是否已经为这个源社区创建了投影到当前层级的社区
                    projection_key = (source_community_id, level)

                    if projection_key not in projected_communities_map:
                        # 创建新的投影社区ID：原始ID + "_0" 表示投影
                        projected_community_id = f"{source_community_id}_0"

                        # 找到源社区
                        source_comm = None
                        for comm in all_communities:
                            if comm.community_id == source_community_id and comm.level == source_level:
                                source_comm = comm
                                break

                        if source_comm:
                            # 创建投影社区（使用新的ID）
                            projected_comm = HierarchicalCommunity(
                                community_id=projected_community_id,
                                level=level,
                                parent_id=source_community_id  # 父社区是原始社区
                            )
                            projected_comm.node_ids = source_comm.node_ids.copy()
                            projected_comm.title = f"Community {source_community_id} (projected to Level {level})"
                            projected_comm.is_projected = True  # 标记为投影社区
                            all_communities.append(projected_comm)

                            # 更新父社区的子社区列表
                            source_comm.children_ids.append(projected_community_id)

                            # 记录投影映射
                            projected_communities_map[projection_key] = projected_community_id

                            logging.debug(f"投影社区 {source_community_id} -> {projected_community_id} (Level {level})")

                    # 更新节点的层级归属（使用投影社区的新ID）
                    node_level_communities[node_name][level] = projected_communities_map[projection_key]

    # Step 5: 对于没有被分配到最底层社区的节点（如果有），分配到其最深层级的社区
    for node_name in node_list:
        if node_name not in node_final_community:
            if node_level_communities[node_name]:
                max_level_assigned = max(node_level_communities[node_name].keys())
                node_final_community[node_name] = node_level_communities[node_name][max_level_assigned]
            else:
                # 极端情况：节点没有被分配到任何社区，分配到根社区
                node_final_community[node_name] = "0"
                node_level_communities[node_name][0] = "0"
    
    # 构建返回结果
    communities_list = [comm.to_dict() for comm in all_communities]
    
    # 构建节点社区映射
    node_community_map = {}
    for node_name in node_list:
        node_community_map[node_name] = {
            "community": node_final_community.get(node_name, "0"),
            "community_levels": {
                f"level_{level}": comm_id 
                for level, comm_id in node_level_communities[node_name].items()
            }
        }
    
    logging.info(f"✅ 递归社区发现完成！共生成 {len(all_communities)} 个社区层级")
    logging.info(f"   层级分布: {_get_level_distribution(all_communities)}")
    
    return communities_list, node_community_map


def _get_level_distribution(communities: List[HierarchicalCommunity]) -> Dict[int, int]:
    """获取各层级的社区数量分布"""
    distribution = {}
    for comm in communities:
        level = comm.level
        distribution[level] = distribution.get(level, 0) + 1
    return distribution


def apply_hierarchical_communities_to_graph(
    G: nx.Graph,
    node_community_map: Dict[str, Any]
) -> nx.Graph:
    """
    将递归社区发现的结果应用到图上
    
    Args:
        G: NetworkX 图对象
        node_community_map: 节点社区映射
    
    Returns:
        更新后的图对象
    """
    logging.info("将分层社区信息写入图节点属性...")
    
    for node_name, comm_info in node_community_map.items():
        if G.has_node(node_name):
            G.nodes[node_name]["community"] = comm_info["community"]
            G.nodes[node_name]["community_levels"] = comm_info["community_levels"]
    
    logging.info("✅ 社区信息已写入图节点")
    return G


def communities_to_dataframe(communities_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    将社区列表转换为 DataFrame
    
    Args:
        communities_list: 社区信息列表
    
    Returns:
        社区信息 DataFrame
    """
    if not communities_list:
        return pd.DataFrame(columns=[
            "community_id", "level", "title", "parent_id", 
            "children_ids", "node_ids", "node_count"
        ])
    
    df = pd.DataFrame(communities_list)
    # 添加节点数量列
    df["node_count"] = df["node_ids"].apply(len)
    
    return df


def save_hierarchical_communities(
    communities_list: List[Dict[str, Any]],
    output_path: str
) -> None:
    """
    保存分层社区信息到文件
    
    Args:
        communities_list: 社区信息列表
        output_path: 输出文件路径（CSV格式）
    """
    import os
    
    df = communities_to_dataframe(communities_list)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存为CSV
    df.to_csv(output_path, index=False, encoding='utf-8')
    logging.info(f"💾 分层社区信息已保存至: {output_path}")


# 示例使用
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # 创建示例图
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
    print("\n社区层级结构:")
    print(df[["community_id", "level", "parent_id", "node_count"]])
    
    # 查看几个节点的社区归属
    print("\n示例节点的社区归属:")
    for node in list(G.nodes())[:5]:
        print(f"节点 {node}:")
        print(f"  最终社区: {G.nodes[node]['community']}")
        print(f"  层级社区: {G.nodes[node]['community_levels']}")
