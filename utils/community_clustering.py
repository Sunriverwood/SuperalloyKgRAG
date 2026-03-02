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
多聚类算法社区发现模块

支持以下聚类算法:
1. Leiden - 基于模块度优化的图聚类算法（默认）
2. Louvain - 经典的模块度优化算法
3. Label Propagation - 标签传播算法
4. Spectral Clustering - 基于图拉普拉斯的谱聚类
5. KMeans - 基于节点嵌入向量的K-Means聚类
6. HDBSCAN - 基于密度的层次聚类
7. Agglomerative - 层次聚合聚类
8. Infomap - 基于信息流的社区发现（igraph）
9. HDBSCAN-Leiden - HDBSCAN 粗聚类 -> 超节点 -> Leiden（二阶段）

对于基于向量的聚类算法（KMeans, HDBSCAN, Agglomerative, Spectral, HDBSCAN-Leiden），
支持从LanceDB中提取节点嵌入向量。
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from pathlib import Path
import networkx as nx
import numpy as np
import lancedb

try:
    import igraph as ig
    import leidenalg as la
    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False
    logging.warning("igraph 或 leidenalg 未安装，Leiden/Louvain 算法不可用")

INFOMAP_AVAILABLE = LEIDEN_AVAILABLE

try:
    from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn 未安装，KMeans/Agglomerative/Spectral 算法不可用")

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    logging.warning("hdbscan 未安装，HDBSCAN 算法不可用")


class ClusteringAlgorithm(Enum):
    """支持的聚类算法枚举"""
    LEIDEN = "leiden"
    LOUVAIN = "louvain"
    LABEL_PROPAGATION = "label_propagation"
    SPECTRAL = "spectral"
    KMEANS = "kmeans"
    HDBSCAN = "hdbscan"
    AGGLOMERATIVE = "agglomerative"
    INFOMAP = "infomap"
    HDBSCAN_LEIDEN = "hdbscan_leiden"


class EmbeddingLoader:
    """
    从LanceDB加载节点嵌入向量
    """

    def __init__(self, db_path: Union[str, Path], table_name: str = "entities"):
        """
        初始化嵌入加载器

        Args:
            db_path: LanceDB数据库路径
            table_name: 实体表名
        """
        self.db_path = Path(db_path)
        self.table_name = table_name
        self._db = None
        self._table = None

    def connect(self):
        """连接到数据库"""
        if self._db is None:
            self._db = lancedb.connect(self.db_path)
            self._table = self._db.open_table(self.table_name)
            logging.info(f"已连接到 LanceDB: {self.db_path}, 表: {self.table_name}")

    def load_embeddings(self, node_ids: List[str]) -> Tuple[np.ndarray, Dict[str, int], List[str]]:
        """
        加载指定节点的嵌入向量

        Args:
            node_ids: 节点ID列表

        Returns:
            embeddings: (N, D) 嵌入矩阵
            node_to_idx: 节点ID到索引的映射
            valid_nodes: 有效节点列表（有嵌入向量的节点）
        """
        self.connect()

        # 获取所有嵌入
        results = self._table.to_pandas()

        # 构建ID到嵌入的映射
        id_to_embedding = {}
        embed_dim = None

        for _, row in results.iterrows():
            node_id = str(row['id'])
            if 'vector' in row:
                embedding = np.array(row['vector'])
                id_to_embedding[node_id] = embedding
                if embed_dim is None:
                    embed_dim = len(embedding)

        logging.info(f"从数据库加载了 {len(id_to_embedding)} 个节点的嵌入向量，维度: {embed_dim}")

        # 对齐嵌入与节点ID
        embeddings = []
        valid_nodes = []
        node_to_idx = {}

        for node_id in node_ids:
            if node_id in id_to_embedding:
                node_to_idx[node_id] = len(valid_nodes)
                valid_nodes.append(node_id)
                embeddings.append(id_to_embedding[node_id])

        if len(valid_nodes) < len(node_ids):
            missing = len(node_ids) - len(valid_nodes)
            logging.warning(f"{missing} 个节点缺少嵌入向量，将被排除在基于向量的聚类之外")

        embeddings = np.array(embeddings) if embeddings else np.array([]).reshape(0, embed_dim or 768)

        return embeddings, node_to_idx, valid_nodes


class CommunityDetector:
    """
    多算法社区发现器

    支持基于图结构的算法和基于向量的算法。
    """

    def __init__(self,
                 algorithm: Union[str, ClusteringAlgorithm] = ClusteringAlgorithm.LEIDEN,
                 embedding_db_path: Optional[Union[str, Path]] = None,
                 embedding_table: str = "entities",
                 **kwargs):
        """
        初始化社区发现器

        Args:
            algorithm: 聚类算法名称或枚举
            embedding_db_path: LanceDB数据库路径（用于基于向量的算法）
            embedding_table: 嵌入表名
            **kwargs: 算法特定参数
        """
        if isinstance(algorithm, str):
            algorithm = ClusteringAlgorithm(algorithm.lower())
        self.algorithm = algorithm
        self.embedding_db_path = embedding_db_path
        self.embedding_table = embedding_table
        self.params = kwargs

        # 验证算法可用性
        self._validate_algorithm()

        # 初始化嵌入加载器（如果需要）
        self.embedding_loader = None
        if self._requires_embeddings() and embedding_db_path:
            self.embedding_loader = EmbeddingLoader(embedding_db_path, embedding_table)

    def _validate_algorithm(self):
        """验证算法是否可用"""
        if self.algorithm in [ClusteringAlgorithm.LEIDEN, ClusteringAlgorithm.LOUVAIN]:
            if not LEIDEN_AVAILABLE:
                raise ImportError(f"算法 {self.algorithm.value} 需要安装 igraph 和 leidenalg")

        if self.algorithm in [ClusteringAlgorithm.KMEANS, ClusteringAlgorithm.AGGLOMERATIVE,
                               ClusteringAlgorithm.SPECTRAL]:
            if not SKLEARN_AVAILABLE:
                raise ImportError(f"算法 {self.algorithm.value} 需要安装 scikit-learn")

        if self.algorithm == ClusteringAlgorithm.HDBSCAN:
            if not HDBSCAN_AVAILABLE:
                raise ImportError("HDBSCAN 算法需要安装 hdbscan 包")

        if self.algorithm == ClusteringAlgorithm.INFOMAP:
            if not INFOMAP_AVAILABLE:
                raise ImportError("Infomap 算法需要安装 igraph 包")

        if self.algorithm == ClusteringAlgorithm.HDBSCAN_LEIDEN:
            if not HDBSCAN_AVAILABLE or not LEIDEN_AVAILABLE:
                raise ImportError("HDBSCAN-Leiden 算法需要安装 hdbscan、igraph 和 leidenalg")

    def _requires_embeddings(self) -> bool:
        """检查算法是否需要嵌入向量"""
        return self.algorithm in [
            ClusteringAlgorithm.KMEANS,
            ClusteringAlgorithm.HDBSCAN,
            ClusteringAlgorithm.AGGLOMERATIVE,
            ClusteringAlgorithm.SPECTRAL,
            ClusteringAlgorithm.HDBSCAN_LEIDEN
        ]

    def detect(self, graph: nx.Graph) -> Tuple[Dict[str, str], List[List[str]]]:
        """
        执行社区发现

        Args:
            graph: NetworkX图（有向或无向）

        Returns:
            community_map: 节点ID到社区ID的映射
            communities: 社区列表，每个社区是节点ID列表
        """
        # 确保是无向图
        if isinstance(graph, nx.DiGraph):
            UG = graph.to_undirected()
        else:
            UG = graph

        node_list = list(UG.nodes())

        if len(node_list) == 0:
            logging.warning("图中没有节点")
            return {}, []

        logging.info(f"使用 {self.algorithm.value} 算法进行社区发现，节点数: {len(node_list)}")

        # 根据算法类型选择执行方法
        if self.algorithm == ClusteringAlgorithm.LEIDEN:
            return self._leiden(UG, node_list)
        elif self.algorithm == ClusteringAlgorithm.LOUVAIN:
            return self._louvain(UG, node_list)
        elif self.algorithm == ClusteringAlgorithm.LABEL_PROPAGATION:
            return self._label_propagation(UG, node_list)
        elif self.algorithm == ClusteringAlgorithm.SPECTRAL:
            return self._spectral(UG, node_list)
        elif self.algorithm == ClusteringAlgorithm.KMEANS:
            return self._kmeans(UG, node_list)
        elif self.algorithm == ClusteringAlgorithm.HDBSCAN:
            return self._hdbscan(UG, node_list)
        elif self.algorithm == ClusteringAlgorithm.AGGLOMERATIVE:
            return self._agglomerative(UG, node_list)
        elif self.algorithm == ClusteringAlgorithm.INFOMAP:
            return self._infomap(UG, node_list)
        elif self.algorithm == ClusteringAlgorithm.HDBSCAN_LEIDEN:
            return self._hdbscan_leiden(UG, node_list)
        else:
            raise ValueError(f"未知的聚类算法: {self.algorithm}")

    def _leiden(self, UG: nx.Graph, node_list: List[str]) -> Tuple[Dict[str, str], List[List[str]]]:
        """Leiden算法"""
        ig_graph = ig.Graph.from_networkx(UG)

        # 获取分辨率参数
        resolution = self.params.get('resolution', 1.0)

        # 当 resolution == 1.0 时使用 ModularityVertexPartition（无需 resolution_parameter）
        # 否则使用 RBConfigurationVertexPartition（支持 resolution_parameter）
        if resolution == 1.0:
            partition = la.find_partition(
                ig_graph,
                la.ModularityVertexPartition
            )
        else:
            partition = la.find_partition(
                ig_graph,
                la.RBConfigurationVertexPartition,
                resolution_parameter=resolution
            )

        node_mapping = {i: name for i, name in enumerate(ig_graph.vs["_nx_name"])}

        community_map = {}
        communities = []

        for comm_id, community in enumerate(partition):
            nodes = [node_mapping[idx] for idx in community]
            communities.append(nodes)
            for node in nodes:
                community_map[node] = str(comm_id)

        logging.info(f"Leiden 算法发现 {len(communities)} 个社区")
        return community_map, communities

    def _louvain(self, UG: nx.Graph, node_list: List[str]) -> Tuple[Dict[str, str], List[List[str]]]:
        """Louvain算法"""
        ig_graph = ig.Graph.from_networkx(UG)

        resolution = self.params.get('resolution', 1.0)

        # 当 resolution == 1.0 时使用 ModularityVertexPartition（无需 resolution_parameter）
        # 否则使用 RBConfigurationVertexPartition（支持 resolution_parameter）
        if resolution == 1.0:
            partition = la.find_partition(
                ig_graph,
                la.ModularityVertexPartition,
                n_iterations=-1  # Louvain 风格的迭代
            )
        else:
            partition = la.find_partition(
                ig_graph,
                la.RBConfigurationVertexPartition,
                resolution_parameter=resolution,
                n_iterations=-1  # Louvain 风格的迭代
            )

        node_mapping = {i: name for i, name in enumerate(ig_graph.vs["_nx_name"])}

        community_map = {}
        communities = []

        for comm_id, community in enumerate(partition):
            nodes = [node_mapping[idx] for idx in community]
            communities.append(nodes)
            for node in nodes:
                community_map[node] = str(comm_id)

        logging.info(f"Louvain 算法发现 {len(communities)} 个社区")
        return community_map, communities

    def _label_propagation(self, UG: nx.Graph, node_list: List[str]) -> Tuple[Dict[str, str], List[List[str]]]:
        """标签传播算法"""
        # 使用 NetworkX 内置的标签传播
        from networkx.algorithms.community import label_propagation_communities

        communities_generator = label_propagation_communities(UG)
        communities = [list(c) for c in communities_generator]

        community_map = {}
        for comm_id, nodes in enumerate(communities):
            for node in nodes:
                community_map[node] = str(comm_id)

        logging.info(f"Label Propagation 算法发现 {len(communities)} 个社区")
        return community_map, communities

    def _spectral(self, UG: nx.Graph, node_list: List[str]) -> Tuple[Dict[str, str], List[List[str]]]:
        """
        谱聚类算法

        可以基于图的邻接矩阵或节点嵌入向量进行聚类
        """
        n_clusters = self.params.get('n_clusters', None)
        use_embeddings = self.params.get('use_embeddings', True)

        if use_embeddings and self.embedding_loader:
            # 基于嵌入向量的谱聚类
            embeddings, node_to_idx, valid_nodes = self.embedding_loader.load_embeddings(node_list)

            if len(valid_nodes) == 0:
                logging.warning("没有有效的嵌入向量，回退到基于图的谱聚类")
                use_embeddings = False
            else:
                if n_clusters is None:
                    # 自动估计簇数（使用sqrt规则）
                    n_clusters = max(2, min(int(np.sqrt(len(valid_nodes))), 100))

                n_clusters = min(n_clusters, len(valid_nodes))

                clustering = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity='nearest_neighbors',
                    n_neighbors=min(10, len(valid_nodes) - 1),
                    assign_labels='kmeans',
                    random_state=42
                )

                labels = clustering.fit_predict(embeddings)

                # 构建结果
                community_map = {}
                communities_dict = {}

                for node, label in zip(valid_nodes, labels):
                    community_map[node] = str(label)
                    if label not in communities_dict:
                        communities_dict[label] = []
                    communities_dict[label].append(node)

                # 处理没有嵌入的节点（分配到最近邻居的社区）
                for node in node_list:
                    if node not in community_map:
                        neighbors = list(UG.neighbors(node))
                        assigned = False
                        for neighbor in neighbors:
                            if neighbor in community_map:
                                community_map[node] = community_map[neighbor]
                                communities_dict[int(community_map[neighbor])].append(node)
                                assigned = True
                                break
                        if not assigned:
                            # 创建单独的社区
                            new_id = max(communities_dict.keys()) + 1 if communities_dict else 0
                            community_map[node] = str(new_id)
                            communities_dict[new_id] = [node]

                communities = list(communities_dict.values())
                logging.info(f"Spectral (向量) 算法发现 {len(communities)} 个社区")
                return community_map, communities

        # 基于图邻接矩阵的谱聚类
        if n_clusters is None:
            n_clusters = max(2, min(int(np.sqrt(len(node_list))), 100))

        n_clusters = min(n_clusters, len(node_list))

        # 构建邻接矩阵
        adj_matrix = nx.to_numpy_array(UG, nodelist=node_list)

        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            assign_labels='kmeans',
            random_state=42
        )

        # 将邻接矩阵转换为相似度矩阵
        similarity = adj_matrix + np.eye(len(node_list))
        labels = clustering.fit_predict(similarity)

        community_map = {}
        communities_dict = {}

        for node, label in zip(node_list, labels):
            community_map[node] = str(label)
            if label not in communities_dict:
                communities_dict[label] = []
            communities_dict[label].append(node)

        communities = list(communities_dict.values())
        logging.info(f"Spectral (图) 算法发现 {len(communities)} 个社区")
        return community_map, communities

    def _kmeans(self, UG: nx.Graph, node_list: List[str]) -> Tuple[Dict[str, str], List[List[str]]]:
        """
        K-Means聚类（基于节点嵌入向量）
        """
        if not self.embedding_loader:
            raise ValueError("KMeans 算法需要提供 embedding_db_path")

        embeddings, node_to_idx, valid_nodes = self.embedding_loader.load_embeddings(node_list)

        if len(valid_nodes) == 0:
            raise ValueError("没有节点具有嵌入向量，无法使用 KMeans")

        n_clusters = self.params.get('n_clusters', None)
        if n_clusters is None:
            # 自动估计簇数
            n_clusters = max(2, min(int(np.sqrt(len(valid_nodes))), 100))

        n_clusters = min(n_clusters, len(valid_nodes))

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300
        )

        labels = kmeans.fit_predict(embeddings)

        # 构建结果
        community_map = {}
        communities_dict = {}

        for node, label in zip(valid_nodes, labels):
            community_map[node] = str(label)
            if label not in communities_dict:
                communities_dict[label] = []
            communities_dict[label].append(node)

        # 处理没有嵌入的节点
        for node in node_list:
            if node not in community_map:
                neighbors = list(UG.neighbors(node))
                assigned = False
                for neighbor in neighbors:
                    if neighbor in community_map:
                        community_map[node] = community_map[neighbor]
                        communities_dict[int(community_map[neighbor])].append(node)
                        assigned = True
                        break
                if not assigned:
                    new_id = max(communities_dict.keys()) + 1 if communities_dict else 0
                    community_map[node] = str(new_id)
                    communities_dict[new_id] = [node]

        communities = list(communities_dict.values())
        logging.info(f"KMeans 算法发现 {len(communities)} 个社区")
        return community_map, communities

    def _hdbscan(self, UG: nx.Graph, node_list: List[str]) -> Tuple[Dict[str, str], List[List[str]]]:
        """
        HDBSCAN 密度聚类（基于节点嵌入向量）
        """
        if not self.embedding_loader:
            raise ValueError("HDBSCAN 算法需要提供 embedding_db_path")

        embeddings, node_to_idx, valid_nodes = self.embedding_loader.load_embeddings(node_list)

        if len(valid_nodes) == 0:
            raise ValueError("没有节点具有嵌入向量，无法使用 HDBSCAN")

        min_cluster_size = self.params.get('min_cluster_size', max(5, len(valid_nodes) // 100))
        min_samples = self.params.get('min_samples', None)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom'
        )

        labels = clusterer.fit_predict(embeddings)

        # HDBSCAN 可能产生噪声点（标签为-1）
        community_map = {}
        communities_dict = {}
        noise_nodes = []

        for node, label in zip(valid_nodes, labels):
            if label == -1:
                noise_nodes.append(node)
            else:
                community_map[node] = str(label)
                if label not in communities_dict:
                    communities_dict[label] = []
                communities_dict[label].append(node)

        # 处理噪声节点：分配到最近邻居的社区或创建单独社区
        for node in noise_nodes:
            neighbors = list(UG.neighbors(node))
            assigned = False
            for neighbor in neighbors:
                if neighbor in community_map:
                    community_map[node] = community_map[neighbor]
                    communities_dict[int(community_map[neighbor])].append(node)
                    assigned = True
                    break
            if not assigned:
                new_id = max(communities_dict.keys()) + 1 if communities_dict else 0
                community_map[node] = str(new_id)
                communities_dict[new_id] = [node]

        # 处理没有嵌入的节点
        for node in node_list:
            if node not in community_map:
                neighbors = list(UG.neighbors(node))
                assigned = False
                for neighbor in neighbors:
                    if neighbor in community_map:
                        community_map[node] = community_map[neighbor]
                        communities_dict[int(community_map[neighbor])].append(node)
                        assigned = True
                        break
                if not assigned:
                    new_id = max(communities_dict.keys()) + 1 if communities_dict else 0
                    community_map[node] = str(new_id)
                    communities_dict[new_id] = [node]

        communities = list(communities_dict.values())
        logging.info(f"HDBSCAN 算法发现 {len(communities)} 个社区（噪声节点: {len(noise_nodes)}）")
        return community_map, communities

    def _agglomerative(self, UG: nx.Graph, node_list: List[str]) -> Tuple[Dict[str, str], List[List[str]]]:
        """
        层次聚合聚类（基于节点嵌入向量）
        """
        if not self.embedding_loader:
            raise ValueError("Agglomerative 算法需要提供 embedding_db_path")

        embeddings, node_to_idx, valid_nodes = self.embedding_loader.load_embeddings(node_list)

        if len(valid_nodes) == 0:
            raise ValueError("没有节点具有嵌入向量，无法使用 Agglomerative")

        n_clusters = self.params.get('n_clusters', None)
        linkage = self.params.get('linkage', 'ward')
        distance_threshold = self.params.get('distance_threshold', None)

        if n_clusters is None and distance_threshold is None:
            n_clusters = max(2, min(int(np.sqrt(len(valid_nodes))), 100))

        if n_clusters is not None:
            n_clusters = min(n_clusters, len(valid_nodes))

        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            distance_threshold=distance_threshold
        )

        labels = clustering.fit_predict(embeddings)

        # 构建结果
        community_map = {}
        communities_dict = {}

        for node, label in zip(valid_nodes, labels):
            community_map[node] = str(label)
            if label not in communities_dict:
                communities_dict[label] = []
            communities_dict[label].append(node)

        # 处理没有嵌入的节点
        for node in node_list:
            if node not in community_map:
                neighbors = list(UG.neighbors(node))
                assigned = False
                for neighbor in neighbors:
                    if neighbor in community_map:
                        community_map[node] = community_map[neighbor]
                        communities_dict[int(community_map[neighbor])].append(node)
                        assigned = True
                        break
                if not assigned:
                    new_id = max(communities_dict.keys()) + 1 if communities_dict else 0
                    community_map[node] = str(new_id)
                    communities_dict[new_id] = [node]

        communities = list(communities_dict.values())
        logging.info(f"Agglomerative 算法发现 {len(communities)} 个社区")
        return community_map, communities

    def _infomap(self, UG: nx.Graph, node_list: List[str]) -> Tuple[Dict[str, str], List[List[str]]]:
        """Infomap算法 - 基于信息流的社区发现（igraph）"""
        if not INFOMAP_AVAILABLE:
            raise ImportError("Infomap 算法需要安装 igraph 包")

        ig_graph = ig.Graph.from_networkx(UG)
        edge_weights = ig_graph.es["weight"] if "weight" in ig_graph.es.attributes() else None

        partition = ig_graph.community_infomap(edge_weights=edge_weights)
        membership = partition.membership
        node_mapping = {i: name for i, name in enumerate(ig_graph.vs["_nx_name"])}

        community_map = {}
        communities_dict = {}

        for idx, module_id in enumerate(membership):
            node = node_mapping[idx]
            community_map[node] = str(module_id)
            if module_id not in communities_dict:
                communities_dict[module_id] = []
            communities_dict[module_id].append(node)

        communities = list(communities_dict.values())
        logging.info(f"Infomap 算法发现 {len(communities)} 个社区")
        return community_map, communities

    def _hdbscan_leiden(self, UG: nx.Graph, node_list: List[str]) -> Tuple[Dict[str, str], List[List[str]]]:
        """HDBSCAN（粗）-> 超节点 -> Leiden（二阶段）"""
        if not self.embedding_loader:
            raise ValueError("HDBSCAN-Leiden 算法需要提供 embedding_db_path")

        embeddings, _, valid_nodes = self.embedding_loader.load_embeddings(node_list)

        if len(valid_nodes) == 0:
            raise ValueError("没有节点具有嵌入向量，无法使用 HDBSCAN-Leiden")

        min_cluster_size = self.params.get('min_cluster_size', max(5, len(valid_nodes) // 100))
        min_samples = self.params.get('min_samples', None)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom'
        )

        labels = clusterer.fit_predict(embeddings)

        # 归一化标签并处理噪声
        label_map = {}
        next_cluster_id = 0
        coarse_map: Dict[str, int] = {}

        for node, label in zip(valid_nodes, labels):
            if label == -1:
                coarse_map[node] = next_cluster_id
                next_cluster_id += 1
                continue
            if label not in label_map:
                label_map[label] = next_cluster_id
                next_cluster_id += 1
            coarse_map[node] = label_map[label]

        # 对没有嵌入的节点，先作为独立粗社区
        for node in node_list:
            if node not in coarse_map:
                coarse_map[node] = next_cluster_id
                next_cluster_id += 1

        # 构建超节点图（社区为节点，跨社区边聚合为权重）
        super_graph = nx.Graph()
        cluster_sizes: Dict[int, int] = {}
        for node, cluster_id in coarse_map.items():
            cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1

        for cluster_id, size in cluster_sizes.items():
            super_graph.add_node(cluster_id, size=size)

        for u, v, data in UG.edges(data=True):
            cu = coarse_map[u]
            cv = coarse_map[v]
            if cu == cv:
                continue
            weight = data.get('weight', 1.0)
            if super_graph.has_edge(cu, cv):
                super_graph[cu][cv]["weight"] += weight
            else:
                super_graph.add_edge(cu, cv, weight=weight)

        if super_graph.number_of_nodes() == 0:
            return {}, []

        # 第二阶段 Leiden
        ig_graph = ig.Graph.from_networkx(super_graph)
        edge_weights = ig_graph.es["weight"] if "weight" in ig_graph.es.attributes() else None
        resolution = self.params.get("leiden_resolution", self.params.get("resolution", 1.0))

        if resolution == 1.0:
            partition = la.find_partition(
                ig_graph,
                la.ModularityVertexPartition,
                weights=edge_weights
            )
        else:
            partition = la.find_partition(
                ig_graph,
                la.RBConfigurationVertexPartition,
                resolution_parameter=resolution,
                weights=edge_weights
            )

        supernode_mapping = {i: name for i, name in enumerate(ig_graph.vs["_nx_name"])}
        supernode_to_comm: Dict[int, int] = {}
        for comm_id, community in enumerate(partition):
            for idx in community:
                cluster_id = supernode_mapping[idx]
                supernode_to_comm[int(cluster_id)] = comm_id

        community_map: Dict[str, str] = {}
        communities_dict: Dict[int, List[str]] = {}

        for node, cluster_id in coarse_map.items():
            comm_id = supernode_to_comm.get(cluster_id, 0)
            community_map[node] = str(comm_id)
            if comm_id not in communities_dict:
                communities_dict[comm_id] = []
            communities_dict[comm_id].append(node)

        communities = list(communities_dict.values())
        logging.info(
            f"HDBSCAN-Leiden 完成: 粗社区 {len(cluster_sizes)} 个 -> 最终社区 {len(communities)} 个"
        )
        return community_map, communities


def detect_communities_with_algorithm(
    graph: nx.DiGraph,
    algorithm: Union[str, ClusteringAlgorithm] = "leiden",
    weight_alpha: float = 0.8,
    embedding_db_path: Optional[Union[str, Path]] = None,
    embedding_table: str = "entities",
    **algorithm_params
) -> Tuple[nx.DiGraph, List[Dict[str, Any]]]:
    """
    使用指定算法执行社区发现（兼容现有接口）

    Args:
        graph: 输入的有向图
        algorithm: 聚类算法名称
        weight_alpha: 社区重要性权重参数
        embedding_db_path: LanceDB数据库路径（用于基于向量的算法）
        embedding_table: 嵌入表名
        **algorithm_params: 算法特定参数
            - resolution: Leiden/Louvain 分辨率参数
            - n_clusters: 簇数量（KMeans/Spectral/Agglomerative）
            - min_cluster_size: HDBSCAN 最小簇大小
            - linkage: Agglomerative 链接方式
            - use_embeddings: Spectral 是否使用嵌入向量

    Returns:
        (图对象, 社区列表)
    """
    import utils.community_importance as ci

    logging.info(f"开始执行社区发现，算法: {algorithm}")

    if not graph.edges:
        logging.warning("图中没有边，无法进行社区发现。")
        return graph, []

    # 创建社区发现器
    detector = CommunityDetector(
        algorithm=algorithm,
        embedding_db_path=embedding_db_path,
        embedding_table=embedding_table,
        **algorithm_params
    )

    # 执行社区发现
    community_map, communities = detector.detect(graph)

    # 将社区ID写入图节点
    nx.set_node_attributes(graph, community_map, "community")

    # 计算节点度数
    UG = graph.to_undirected()
    for node in graph.nodes():
        graph.nodes[node]["degree"] = UG.degree(node)

    # 计算社区重要性
    if communities:
        ci.calculate_community_importance(graph, communities, weight_alpha=weight_alpha)

    logging.info("已将社区ID、节点degree和关系重要性分数标注到图谱。")

    # 构建扁平社区列表（与现有格式兼容）
    communities_list = []
    for idx, nodes in enumerate(communities):
        communities_list.append({
            "community_id": str(idx),
            "level": 0,
            "title": f"Community {idx}",
            "parent_id": None,
            "children_ids": [],
            "node_ids": nodes
        })

    logging.info(f"社区发现完成！共发现 {len(communities)} 个社区。")

    return graph, communities_list


def get_available_algorithms() -> List[str]:
    """获取可用的聚类算法列表"""
    available = []

    if LEIDEN_AVAILABLE:
        available.extend(["leiden", "louvain"])

    # Label Propagation 使用 NetworkX 内置实现
    available.append("label_propagation")

    if SKLEARN_AVAILABLE:
        available.extend(["kmeans", "spectral", "agglomerative"])

    if HDBSCAN_AVAILABLE:
        available.append("hdbscan")

    if INFOMAP_AVAILABLE:
        available.append("infomap")

    if HDBSCAN_AVAILABLE and LEIDEN_AVAILABLE:
        available.append("hdbscan_leiden")

    return available


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    print("可用的聚类算法:")
    for algo in get_available_algorithms():
        print(f"  - {algo}")

    # 创建测试图
    G = nx.karate_club_graph()
    G = G.to_directed()

    print(f"\n测试图: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")

    # 测试各种算法
    for algo in ["leiden", "louvain", "label_propagation", "infomap", "hdbscan_leiden"]:
        if algo in get_available_algorithms():
            print(f"\n测试 {algo} 算法...")
            detector = CommunityDetector(algorithm=algo)
            community_map, communities = detector.detect(G)
            print(f"  发现 {len(communities)} 个社区")

