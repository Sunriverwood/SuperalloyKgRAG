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
Data Loader for Graph Reasoning System

Loads knowledge graph structure and embeddings from:
- final_graph.json (NetworkX graph with nodes, edges, relationships)
- embedding.db (LanceDB with entity and relationship embeddings)
"""

import json
import logging
import numpy as np
import networkx as nx
import lancedb
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class GraphData:
    """Container for graph structure and embeddings"""
    # Graph structure
    G: nx.DiGraph
    node_ids: List[str]
    edge_index: torch.Tensor  # [2, num_edges]
    edge_types: torch.Tensor  # [num_edges]
    edge_weights: torch.Tensor  # [num_edges] - from composite_importance

    # Embeddings
    node_embeddings: torch.Tensor  # [num_nodes, embed_dim]
    edge_type_embeddings: torch.Tensor  # [num_edge_types, embed_dim]

    # Mappings
    node_to_idx: Dict[str, int]
    idx_to_node: Dict[int, str]
    edge_type_to_idx: Dict[str, int]
    idx_to_edge_type: Dict[int, str]

    # Metadata
    num_nodes: int
    num_edges: int
    num_edge_types: int
    embed_dim: int

    # Adjacency mask for graph constraints
    adjacency_mask: Optional[torch.Tensor] = None  # [num_nodes, num_nodes]


class GraphReasoningDataLoader:
    """
    Loads and prepares graph data for reasoning tasks.

    Responsibilities:
    1. Load NetworkX graph from final_graph.json
    2. Extract embeddings from LanceDB
    3. Build edge_index, edge_types, edge_weights
    4. Create adjacency mask for graph constraints
    5. Normalize embeddings and weights
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.reasoning_config = config.get("reasoning", {})
        self.data_config = self.reasoning_config.get("data", {})

        # Paths
        from pathlib import Path
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        self.graph_path = PROJECT_ROOT / self.data_config.get("graph_path", "data/graphs/final_graph.json")
        self.embedding_db_path = PROJECT_ROOT / self.data_config.get("entity_embeddings_db", "data/embeddings/enriched.db")
        self.entity_table = self.data_config.get("entity_table", "entities")
        self.relationship_table = self.data_config.get("relationship_table", "relationships")

        logging.info(f"GraphReasoningDataLoader initialized")
        logging.info(f"  Graph path: {self.graph_path}")
        logging.info(f"  Embedding DB: {self.embedding_db_path}")

    def load_graph(self) -> nx.DiGraph:
        """Load NetworkX graph from final_graph.json"""
        logging.info(f"Loading graph from {self.graph_path}...")

        with open(self.graph_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)

        G = nx.node_link_graph(graph_data, directed=True, edges="links")

        logging.info(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G

    def load_embeddings(self, node_ids: List[str]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Load embeddings from LanceDB.

        Returns:
            node_embeddings: [num_nodes, embed_dim]
            edge_type_embeddings: dict mapping edge_type -> embedding
        """
        logging.info(f"Loading embeddings from {self.embedding_db_path}...")

        db = lancedb.connect(self.embedding_db_path)

        # Load entity embeddings
        entity_table = db.open_table(self.entity_table)

        # Create mapping for quick lookup
        node_embeddings_dict = {}
        embed_dim = None

        # Fetch all embeddings
        results = entity_table.to_pandas()
        for _, row in results.iterrows():
            node_id = str(row['id'])
            if 'vector' in row:
                embedding = np.array(row['vector'])
                node_embeddings_dict[node_id] = embedding
                if embed_dim is None:
                    embed_dim = len(embedding)

        logging.info(f"Loaded {len(node_embeddings_dict)} entity embeddings, dim={embed_dim}")

        # Align embeddings with node_ids
        node_embeddings = []
        missing_count = 0
        for node_id in node_ids:
            if node_id in node_embeddings_dict:
                node_embeddings.append(node_embeddings_dict[node_id])
            else:
                # Initialize missing nodes with zero vector
                node_embeddings.append(np.zeros(embed_dim))
                missing_count += 1

        if missing_count > 0:
            logging.warning(f"{missing_count} nodes missing embeddings, initialized with zeros")

        node_embeddings = np.array(node_embeddings)

        # Load relationship embeddings (if available)
        edge_type_embeddings = {}
        try:
            rel_table = db.open_table(self.relationship_table)
            rel_results = rel_table.to_pandas()

            for _, row in rel_results.iterrows():
                edge_type = str(row.get('description', row.get('relation', 'unknown')))
                if 'vector' in row:
                    edge_type_embeddings[edge_type] = np.array(row['vector'])

            logging.info(f"Loaded {len(edge_type_embeddings)} relationship type embeddings")
        except Exception as e:
            logging.warning(f"Could not load relationship embeddings: {e}")

        return node_embeddings, edge_type_embeddings

    def build_edge_index_and_weights(self, G: nx.DiGraph, node_to_idx: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
        """
        Build edge_index, edge_types, and edge_weights from graph.

        Returns:
            edge_index: [2, num_edges] array of (source, target) indices
            edge_types: [num_edges] array of edge type indices
            edge_weights: [num_edges] array of composite_importance or weight values
            edge_type_to_idx: mapping from edge type string to index
        """
        logging.info("Building edge index and extracting edge weights...")

        edge_list = []
        edge_type_list = []
        edge_weight_list = []
        edge_type_set = set()

        for u, v, data in G.edges(data=True):
            if u not in node_to_idx or v not in node_to_idx:
                continue

            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]

            # Extract edge type (relationship description)
            edge_type = data.get('description') or data.get('relationship') or data.get('relation') or 'unknown'
            edge_type_set.add(edge_type)

            # Extract edge weight (prioritize composite_importance, fallback to weight)
            edge_weight = data.get('composite_importance', data.get('weight', 1.0))
            try:
                edge_weight = float(edge_weight)
            except (TypeError, ValueError):
                edge_weight = 1.0

            edge_list.append([u_idx, v_idx])
            edge_type_list.append(edge_type)
            edge_weight_list.append(edge_weight)

        # Create edge type mapping
        edge_type_to_idx = {et: idx for idx, et in enumerate(sorted(edge_type_set))}
        edge_type_indices = [edge_type_to_idx[et] for et in edge_type_list]

        edge_index = np.array(edge_list).T  # [2, num_edges]
        edge_types = np.array(edge_type_indices)
        edge_weights = np.array(edge_weight_list)

        logging.info(f"Built edge index: {edge_index.shape[1]} edges, {len(edge_type_to_idx)} edge types")
        logging.info(f"Edge weight stats: min={edge_weights.min():.4f}, max={edge_weights.max():.4f}, mean={edge_weights.mean():.4f}")

        return edge_index, edge_types, edge_weights, edge_type_to_idx

    def create_adjacency_mask(self, num_nodes: int, edge_index: np.ndarray) -> np.ndarray:
        """
        Create binary adjacency mask for graph constraints.
        mask[i, j] = 1 if edge (i, j) exists, 0 otherwise.
        """
        logging.info("Creating adjacency mask for graph constraints...")

        mask = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        mask[edge_index[0], edge_index[1]] = 1.0

        num_edges = edge_index.shape[1]
        logging.info(f"Adjacency mask created: {num_nodes}x{num_nodes}, {num_edges} non-zero entries")

        return mask

    def initialize_edge_type_embeddings(self, edge_type_to_idx: Dict[str, int],
                                       edge_type_embeddings_dict: Dict[str, np.ndarray],
                                       embed_dim: int) -> np.ndarray:
        """
        Initialize edge type embeddings matrix.
        Use loaded embeddings if available, otherwise random initialization.
        """
        num_edge_types = len(edge_type_to_idx)
        edge_type_embeddings = np.random.randn(num_edge_types, embed_dim).astype(np.float32) * 0.01

        for edge_type, idx in edge_type_to_idx.items():
            if edge_type in edge_type_embeddings_dict:
                edge_type_embeddings[idx] = edge_type_embeddings_dict[edge_type]

        logging.info(f"Initialized edge type embeddings: {edge_type_embeddings.shape}")
        return edge_type_embeddings

    def load(self, device: str = 'cpu') -> GraphData:
        """
        Main loading function that orchestrates all data preparation.

        Args:
            device: 'cpu' or 'cuda'

        Returns:
            GraphData object containing all necessary data for reasoning
        """
        logging.info("="*60)
        logging.info("Loading Graph Reasoning Data")
        logging.info("="*60)

        # 1. Load graph structure
        G = self.load_graph()

        # 2. Create node mappings
        node_ids = list(G.nodes())
        node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        idx_to_node = {idx: node_id for node_id, idx in node_to_idx.items()}
        num_nodes = len(node_ids)

        # 3. Load embeddings
        node_embeddings_np, edge_type_embeddings_dict = self.load_embeddings(node_ids)
        embed_dim = node_embeddings_np.shape[1]

        # 4. Build edge index and weights
        edge_index_np, edge_types_np, edge_weights_np, edge_type_to_idx = \
            self.build_edge_index_and_weights(G, node_to_idx)

        idx_to_edge_type = {idx: et for et, idx in edge_type_to_idx.items()}
        num_edges = edge_index_np.shape[1]
        num_edge_types = len(edge_type_to_idx)

        # 5. Initialize edge type embeddings
        edge_type_embeddings_np = self.initialize_edge_type_embeddings(
            edge_type_to_idx, edge_type_embeddings_dict, embed_dim
        )

        # 6. Create adjacency mask
        adjacency_mask_np = self.create_adjacency_mask(num_nodes, edge_index_np)

        # 7. Convert to PyTorch tensors
        logging.info(f"Converting to PyTorch tensors on device: {device}")

        node_embeddings = torch.from_numpy(node_embeddings_np).float().to(device)
        edge_index = torch.from_numpy(edge_index_np).long().to(device)
        edge_types = torch.from_numpy(edge_types_np).long().to(device)
        edge_weights = torch.from_numpy(edge_weights_np).float().to(device)
        edge_type_embeddings = torch.from_numpy(edge_type_embeddings_np).float().to(device)
        adjacency_mask = torch.from_numpy(adjacency_mask_np).float().to(device)

        # 8. Create GraphData object
        graph_data = GraphData(
            G=G,
            node_ids=node_ids,
            edge_index=edge_index,
            edge_types=edge_types,
            edge_weights=edge_weights,
            node_embeddings=node_embeddings,
            edge_type_embeddings=edge_type_embeddings,
            node_to_idx=node_to_idx,
            idx_to_node=idx_to_node,
            edge_type_to_idx=edge_type_to_idx,
            idx_to_edge_type=idx_to_edge_type,
            num_nodes=num_nodes,
            num_edges=num_edges,
            num_edge_types=num_edge_types,
            embed_dim=embed_dim,
            adjacency_mask=adjacency_mask
        )

        logging.info("="*60)
        logging.info("Graph Data Loading Complete")
        logging.info(f"  Nodes: {num_nodes}")
        logging.info(f"  Edges: {num_edges}")
        logging.info(f"  Edge Types: {num_edge_types}")
        logging.info(f"  Embedding Dim: {embed_dim}")
        logging.info("="*60)

        return graph_data


if __name__ == "__main__":
    # Test the data loader
    import yaml
    from pathlib import Path

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Load config
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    config_path = PROJECT_ROOT / "config" / "settings.yaml"

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Test loading
    loader = GraphReasoningDataLoader(config)
    graph_data = loader.load(device='cpu')

    print("\n✓ Data loading test successful!")
    print(f"  Node embeddings shape: {graph_data.node_embeddings.shape}")
    print(f"  Edge index shape: {graph_data.edge_index.shape}")
    print(f"  Edge weights shape: {graph_data.edge_weights.shape}")
    print(f"  Adjacency mask shape: {graph_data.adjacency_mask.shape}")

