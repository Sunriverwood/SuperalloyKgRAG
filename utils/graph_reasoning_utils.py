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
Utility Functions for Graph Reasoning

Provides helper functions for:
- Graph constraint application (adjacency masking)
- Edge weight normalization
- Path scoring
- Pseudo query generation
"""

import logging
import numpy as np
import torch
import networkx as nx
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class PathInfo:
    """Information about a reasoning path"""
    path: List[str]  # List of node IDs
    score: float  # Path probability/score
    edge_scores: List[float]  # Individual edge scores
    edge_types: List[str]  # Relationship types along path
    composite_scores: List[float]  # Composite importance scores


def apply_adjacency_mask(scores: torch.Tensor, adjacency_mask: torch.Tensor,
                        fill_value: float = -1e9) -> torch.Tensor:
    """
    Apply graph constraint by masking non-existent edges.

    Args:
        scores: [num_nodes, num_nodes] or [batch, num_nodes, num_nodes] attention scores
        adjacency_mask: [num_nodes, num_nodes] binary mask (1 for existing edges, 0 otherwise)
        fill_value: Value to fill for non-existing edges (default: -1e9 for softmax)

    Returns:
        Masked scores with non-edges set to fill_value
    """
    mask = adjacency_mask.bool()

    if scores.dim() == 2:
        # [num_nodes, num_nodes]
        masked_scores = torch.where(mask, scores, torch.full_like(scores, fill_value))
    elif scores.dim() == 3:
        # [batch, num_nodes, num_nodes]
        mask = mask.unsqueeze(0)  # [1, num_nodes, num_nodes]
        masked_scores = torch.where(mask, scores, torch.full_like(scores, fill_value))
    else:
        raise ValueError(f"Unsupported score tensor dimension: {scores.dim()}")

    return masked_scores


def normalize_edge_weights(G: nx.DiGraph, use_composite: bool = True,
                          strategy: str = 'minmax') -> Dict[Tuple[str, str], float]:
    """
    Normalize edge weights from the graph.

    Args:
        G: NetworkX graph
        use_composite: If True, use 'composite_importance', else use 'weight'
        strategy: Normalization strategy - 'minmax', 'softmax', 'log', or 'none'

    Returns:
        Dictionary mapping (source, target) -> normalized_weight
    """
    weights = []
    edge_list = []

    for u, v, data in G.edges(data=True):
        if use_composite:
            w = data.get('composite_importance', data.get('weight', 1.0))
        else:
            w = data.get('weight', 1.0)

        try:
            w = float(w)
        except (TypeError, ValueError):
            w = 1.0

        weights.append(w)
        edge_list.append((u, v))

    weights = np.array(weights)

    # Apply normalization strategy
    if strategy == 'minmax':
        w_min, w_max = weights.min(), weights.max()
        if w_max > w_min:
            normalized = (weights - w_min) / (w_max - w_min)
        else:
            normalized = np.ones_like(weights)

    elif strategy == 'softmax':
        # Temperature-scaled softmax
        normalized = np.exp(weights) / np.exp(weights).sum()
        normalized = normalized * len(weights)  # Scale back

    elif strategy == 'log':
        # Log-scale normalization
        normalized = np.log1p(weights)
        normalized = normalized / normalized.max() if normalized.max() > 0 else normalized

    elif strategy == 'none':
        normalized = weights

    else:
        raise ValueError(f"Unknown normalization strategy: {strategy}")

    # Create mapping
    weight_dict = {edge: float(norm_w) for edge, norm_w in zip(edge_list, normalized)}

    logging.info(f"Normalized {len(weight_dict)} edge weights using strategy '{strategy}'")
    logging.info(f"  Original range: [{weights.min():.4f}, {weights.max():.4f}]")
    logging.info(f"  Normalized range: [{normalized.min():.4f}, {normalized.max():.4f}]")

    return weight_dict


def score_path_by_importance(path: List[str], G: nx.DiGraph,
                             attention_scores: Optional[Dict[Tuple[str, str], float]] = None,
                             alpha: float = 0.5) -> Tuple[float, List[float], List[float]]:
    """
    Score a path using composite_importance and optional attention scores.

    Args:
        path: List of node IDs forming a path
        G: NetworkX graph
        attention_scores: Optional dict mapping (u, v) -> attention weight
        alpha: Weight for combining importance and attention (0=only importance, 1=only attention)

    Returns:
        total_score: Overall path score (product of edge scores)
        edge_scores: Individual edge scores
        composite_scores: Composite importance values
    """
    if len(path) < 2:
        return 1.0, [], []

    edge_scores = []
    composite_scores = []

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]

        # Check if edge exists
        if not G.has_edge(u, v):
            return 0.0, [0.0] * (len(path) - 1), [0.0] * (len(path) - 1)

        # Get composite importance
        edge_data = G[u][v]
        composite_score = edge_data.get('composite_importance', edge_data.get('weight', 1.0))
        try:
            composite_score = float(composite_score)
        except (TypeError, ValueError):
            composite_score = 1.0

        composite_scores.append(composite_score)

        # Combine with attention if available
        if attention_scores and (u, v) in attention_scores:
            attn_score = attention_scores[(u, v)]
            # Weighted combination
            combined_score = (1 - alpha) * composite_score + alpha * attn_score
        else:
            combined_score = composite_score

        edge_scores.append(combined_score)

    # Path score: geometric mean of edge scores
    # Using geometric mean (prod^(1/n)) instead of raw product to avoid
    # exponential decay on multi-hop paths. This makes the score represent
    # "average edge quality" and is path-length invariant.
    if edge_scores:
        n = len(edge_scores)
        product = np.prod(edge_scores)
        if product > 0:
            total_score = product ** (1.0 / n)
        else:
            total_score = 0.0
    else:
        total_score = 0.0

    return float(total_score), edge_scores, composite_scores


def extract_paths_bfs(G: nx.DiGraph, start_nodes: List[str], end_nodes: List[str],
                     max_depth: int = 3, max_paths: int = 100) -> List[List[str]]:
    """
    Extract paths from start nodes to end nodes using BFS on the real graph.

    Args:
        G: NetworkX graph (only real edges considered)
        start_nodes: List of starting node IDs
        end_nodes: Set of target node IDs
        max_depth: Maximum path length
        max_paths: Maximum number of paths to return

    Returns:
        List of paths, where each path is a list of node IDs
    """
    end_set = set(end_nodes)
    paths = []

    for start in start_nodes:
        if start not in G:
            continue

        # If start node is also an end node, add it as a single-node path
        # (This happens when the query directly matches relevant entities)
        if start in end_set:
            paths.append([start])
            if len(paths) >= max_paths:
                break
            continue

        # BFS with path tracking for multi-hop paths
        queue = [(start, [start])]
        visited_paths = set()

        while queue and len(paths) < max_paths:
            current, path = queue.pop(0)

            # Check if reached target
            if current in end_set:
                paths.append(path)
                if len(paths) >= max_paths:
                    break
                continue

            # Check depth limit
            if len(path) >= max_depth:
                continue

            # Expand to neighbors (only real edges in G)
            for neighbor in G.neighbors(current):
                if neighbor not in path:  # Avoid cycles
                    new_path = path + [neighbor]
                    path_key = tuple(new_path)

                    if path_key not in visited_paths:
                        visited_paths.add(path_key)
                        queue.append((neighbor, new_path))

    logging.info(f"Extracted {len(paths)} paths from {len(start_nodes)} start nodes to {len(end_nodes)} end nodes")

    return paths


def rank_paths(paths: List[List[str]], G: nx.DiGraph,
              attention_scores: Optional[Dict[Tuple[str, str], float]] = None,
              top_k: int = 10, alpha: float = 0.5) -> List[PathInfo]:
    """
    Rank paths by their scores and return top-k.

    Args:
        paths: List of paths (each path is a list of node IDs)
        G: NetworkX graph
        attention_scores: Optional attention weights
        top_k: Number of top paths to return
        alpha: Weight for combining importance and attention

    Returns:
        List of PathInfo objects sorted by score (descending)
    """
    path_infos = []

    for path in paths:
        total_score, edge_scores, composite_scores = score_path_by_importance(
            path, G, attention_scores, alpha
        )

        # Extract edge types
        edge_types = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if G.has_edge(u, v):
                edge_data = G[u][v]
                edge_type = edge_data.get('description', edge_data.get('relationship', 'unknown'))
                edge_types.append(edge_type)
            else:
                edge_types.append('missing')

        path_info = PathInfo(
            path=path,
            score=total_score,
            edge_scores=edge_scores,
            edge_types=edge_types,
            composite_scores=composite_scores
        )

        path_infos.append(path_info)

    # Sort by score
    path_infos.sort(key=lambda x: x.score, reverse=True)

    return path_infos[:top_k]


def format_path_explanation(path_info: PathInfo, G: nx.DiGraph,
                           include_scores: bool = True) -> str:
    """
    Format a path as a human-readable explanation.

    Args:
        path_info: PathInfo object
        G: NetworkX graph
        include_scores: Whether to include numerical scores

    Returns:
        Formatted string representation
    """
    if not path_info.path:
        return "Empty path (no nodes)"

    if len(path_info.path) == 1:
        node_id = path_info.path[0]
        node_name = G.nodes[node_id].get('name', node_id) if node_id in G.nodes else node_id
        return f"Single-node path: {node_name} (no edges)"

    lines = []
    lines.append(f"Path (score: {path_info.score:.4f}):")

    for i in range(len(path_info.path) - 1):
        node_id = path_info.path[i]
        next_node_id = path_info.path[i + 1]

        # Get node names
        node_name = G.nodes[node_id].get('name', node_id) if node_id in G.nodes else node_id
        next_node_name = G.nodes[next_node_id].get('name', next_node_id) if next_node_id in G.nodes else next_node_id

        # Get edge info
        edge_type = path_info.edge_types[i] if i < len(path_info.edge_types) else 'unknown'

        if include_scores:
            edge_score = path_info.edge_scores[i] if i < len(path_info.edge_scores) else 0.0
            composite_score = path_info.composite_scores[i] if i < len(path_info.composite_scores) else 0.0
            lines.append(f"  {node_name} --[{edge_type}]--> {next_node_name}")
            lines.append(f"    (importance: {composite_score:.4f}, score: {edge_score:.4f})")
        else:
            lines.append(f"  {node_name} --[{edge_type}]--> {next_node_name}")

    # Add final node
    final_node_id = path_info.path[-1]
    final_node_name = G.nodes[final_node_id].get('name', final_node_id) if final_node_id in G.nodes else final_node_id
    lines.append(f"  → {final_node_name}")

    return "\n".join(lines)


class PseudoQueryGenerator:
    """
    Generates pseudo queries from graph triplets for self-supervised training.

    Strategy:
    1. For each edge (h, r, t), create pseudo query from (h, r)
    2. Use text descriptions of head entity + relation as query text
    3. Tail entity becomes the positive target
    """

    def __init__(self, G: nx.DiGraph, text_encoder=None):
        """
        Args:
            G: NetworkX graph
            text_encoder: Optional encoder for converting text to embeddings
        """
        self.G = G
        self.text_encoder = text_encoder

    def generate_triplets(self, max_triplets: Optional[int] = None) -> List[Tuple[str, str, str]]:
        """
        Generate (head, relation, tail) triplets from graph.

        Returns:
            List of (head_id, relation, tail_id) tuples
        """
        triplets = []

        for u, v, data in self.G.edges(data=True):
            relation = data.get('description', data.get('relationship', 'related_to'))
            triplets.append((u, relation, v))

        if max_triplets and len(triplets) > max_triplets:
            # Random sample
            import random
            triplets = random.sample(triplets, max_triplets)

        logging.info(f"Generated {len(triplets)} triplets for pseudo query training")
        return triplets

    def triplet_to_query_text(self, head_id: str, relation: str) -> str:
        """
        Convert (head, relation) to query text.

        Args:
            head_id: Head entity node ID
            relation: Relation description

        Returns:
            Query text string
        """
        if head_id in self.G.nodes:
            head_name = self.G.nodes[head_id].get('name', head_id)
            head_desc = self.G.nodes[head_id].get('description', '')

            if head_desc:
                query_text = f"{head_name}: {head_desc}. Relation: {relation}"
            else:
                query_text = f"{head_name}. Relation: {relation}"
        else:
            query_text = f"Entity: {head_id}. Relation: {relation}"

        return query_text

    def encode_query(self, query_text: str) -> Optional[np.ndarray]:
        """
        Encode query text to embedding vector.

        Args:
            query_text: Natural language query

        Returns:
            Query embedding vector or None if no encoder
        """
        if self.text_encoder is None:
            return None

        # This would use the same encoder as used for entity embeddings
        # For now, return None - will be implemented when integrating with actual encoder
        return None


if __name__ == "__main__":
    # Test utilities
    import json

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    print("✓ Graph reasoning utilities module loaded successfully")

