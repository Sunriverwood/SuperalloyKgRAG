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
Graph Reasoning Inference Engine

Provides reasoning capabilities:
1. Node scoring with query-aware embeddings
2. Personalized PageRank (PPR) propagation
3. Path extraction and ranking
"""

import torch
import torch.nn.functional as F
import logging
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

from core.reasoning.models.rgat import QueryAwareRGAT
from core.reasoning.data_loader import GraphData
from utils.graph_reasoning_utils import (
    extract_paths_bfs, rank_paths, PathInfo, format_path_explanation
)


class GraphReasoner:
    """
    Inference engine for query-aware graph reasoning.

    Workflow:
    1. Encode query
    2. Score nodes using trained matching function
    3. Propagate via PPR or query-aware GNN
    4. Extract and rank paths
    """

    def __init__(self, config: Dict[str, Any], graph_data: GraphData,
                 gnn_model: QueryAwareRGAT, query_matcher, device: str = 'cpu'):
        """
        Args:
            config: Configuration dictionary
            graph_data: GraphData object
            gnn_model: Trained QueryAwareRGAT model
            query_matcher: Trained query-entity matcher
            device: 'cpu' or 'cuda'
        """
        self.config = config
        self.reasoning_config = config.get('reasoning', {})
        self.inference_config = self.reasoning_config.get('inference', {})

        self.graph_data = graph_data
        self.gnn = gnn_model.to(device)
        self.query_matcher = query_matcher.to(device)
        self.device = device

        # Set to eval mode
        self.gnn.eval()
        self.query_matcher.eval()

        # PPR configuration
        self.ppr_alpha = self.inference_config.get('ppr_alpha', 0.15)
        self.ppr_max_iter = self.inference_config.get('ppr_max_iter', 100)
        self.ppr_tol = self.inference_config.get('ppr_tol', 1e-6)

        # Path extraction configuration
        self.max_path_length = self.inference_config.get('max_path_length', 3)
        self.max_paths_per_query = self.inference_config.get('max_paths_per_query', 10)
        self.min_path_score = self.inference_config.get('min_path_score', 0.01)
        self.top_k_nodes = self.inference_config.get('top_k_nodes', 20)

        logging.info("GraphReasoner initialized")
        logging.info(f"  PPR alpha: {self.ppr_alpha}")
        logging.info(f"  Max path length: {self.max_path_length}")

    def encode_query(self, query_text: str, text_encoder=None) -> torch.Tensor:
        """
        Encode query text to embedding vector.

        Args:
            query_text: Natural language query
            text_encoder: Text encoding function (if None, returns zero vector)

        Returns:
            Query embedding [embed_dim]
        """
        if text_encoder is not None:
            # Use provided text encoder
            query_emb = text_encoder(query_text)
            return torch.tensor(query_emb, dtype=torch.float32, device=self.device)
        else:
            # Placeholder: return zero vector
            logging.warning("No text encoder provided, using zero vector for query")
            return torch.zeros(self.graph_data.embed_dim, device=self.device)

    @torch.no_grad()
    def score_nodes(self, query_emb: torch.Tensor, use_gnn: bool = True) -> torch.Tensor:
        """
        Score all nodes based on query relevance.

        Args:
            query_emb: Query embedding [embed_dim]
            use_gnn: If True, use query-aware GNN; else use GNN without query

        Returns:
            Node scores [num_nodes]
        """
        # Always use GNN to get correct dimension for query_matcher
        # The difference is whether we pass the query for query-aware attention
        node_embeddings = self.gnn(
            x=self.graph_data.node_embeddings,
            edge_index=self.graph_data.edge_index,
            edge_type_emb=self.graph_data.edge_type_embeddings[self.graph_data.edge_types],
            edge_weights=self.graph_data.edge_weights,
            query_emb=query_emb if use_gnn else None,  # Pass query only if use_gnn=True
            adjacency_mask=self.graph_data.adjacency_mask
        )

        # Compute similarity scores using query matcher
        scores = self.query_matcher(query_emb, node_embeddings)

        return scores

    @torch.no_grad()
    def get_top_k_nodes(self, query_emb: torch.Tensor, k: Optional[int] = None) -> Tuple[List[str], torch.Tensor]:
        """
        Get top-k most relevant nodes for the query.

        Args:
            query_emb: Query embedding
            k: Number of top nodes to return (uses config if None)

        Returns:
            node_ids: List of top-k node IDs
            scores: Scores for top-k nodes
        """
        if k is None:
            k = self.top_k_nodes

        # Score all nodes
        scores = self.score_nodes(query_emb, use_gnn=True)

        # Get top-k
        top_k_scores, top_k_indices = torch.topk(scores, k=min(k, len(scores)))

        # Convert indices to node IDs
        top_k_node_ids = [self.graph_data.idx_to_node[idx.item()] for idx in top_k_indices]

        logging.info(f"Top-{k} nodes retrieved, score range: [{top_k_scores.min():.4f}, {top_k_scores.max():.4f}]")

        return top_k_node_ids, top_k_scores

    def build_transition_matrix(self) -> np.ndarray:
        """
        Build transition matrix for PPR using composite_importance as edge weights.

        Returns:
            Transition matrix P [num_nodes, num_nodes]
            P[i, j] = normalized weight of edge i -> j
        """
        num_nodes = self.graph_data.num_nodes
        P = np.zeros((num_nodes, num_nodes))

        # Use edge weights (composite_importance)
        edge_index = self.graph_data.edge_index.cpu().numpy()
        edge_weights = self.graph_data.edge_weights.cpu().numpy()

        # Fill in edge weights
        for i in range(edge_index.shape[1]):
            src, tgt = edge_index[0, i], edge_index[1, i]
            P[src, tgt] = edge_weights[i]

        # Normalize rows (out-degree normalization)
        row_sums = P.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # Avoid division by zero
        P = P / row_sums

        return P

    @torch.no_grad()
    def propagate_ppr(self, query_emb: torch.Tensor, alpha: Optional[float] = None,
                     max_iter: Optional[int] = None, tol: Optional[float] = None) -> np.ndarray:
        """
        Personalized PageRank propagation with query-aware initialization.

        π_{k+1} = α * π_0 + (1-α) * π_k * P

        Args:
            query_emb: Query embedding
            alpha: Restart probability (uses config if None)
            max_iter: Maximum iterations (uses config if None)
            tol: Convergence tolerance (uses config if None)

        Returns:
            Final PPR scores [num_nodes]
        """
        if alpha is None:
            alpha = self.ppr_alpha
        if max_iter is None:
            max_iter = self.ppr_max_iter
        if tol is None:
            tol = self.ppr_tol

        num_nodes = self.graph_data.num_nodes

        # Initialize with query-based scores
        initial_scores = self.score_nodes(query_emb, use_gnn=False).cpu().numpy()
        initial_scores = np.maximum(initial_scores, 0)  # Ensure non-negative

        # Normalize to probability distribution
        if initial_scores.sum() > 0:
            pi_0 = initial_scores / initial_scores.sum()
        else:
            pi_0 = np.ones(num_nodes) / num_nodes

        # Build transition matrix
        P = self.build_transition_matrix()

        # PPR iteration
        pi = pi_0.copy()

        for iteration in range(max_iter):
            pi_new = alpha * pi_0 + (1 - alpha) * pi @ P

            # Check convergence
            diff = np.abs(pi_new - pi).max()
            pi = pi_new

            if diff < tol:
                logging.info(f"PPR converged at iteration {iteration+1}, diff={diff:.2e}")
                break
        else:
            logging.warning(f"PPR did not converge after {max_iter} iterations")

        return pi

    @torch.no_grad()
    def extract_attention_scores(self, query_emb: torch.Tensor) -> Dict[Tuple[str, str], float]:
        """
        Extract attention scores from GNN for interpretability.

        Args:
            query_emb: Query embedding

        Returns:
            Dictionary mapping (source_id, target_id) -> attention_weight
        """
        attention_dict = {}
        edge_index = self.graph_data.edge_index.cpu().numpy()

        # Use GNN's extract_attention method to get real attention weights
        try:
            _, attentions = self.gnn.extract_attention(
                x=self.graph_data.node_embeddings,
                edge_index=self.graph_data.edge_index,
                edge_type_emb=self.graph_data.edge_type_embeddings[self.graph_data.edge_types],
                edge_weights=self.graph_data.edge_weights,
                query_emb=query_emb,
                adjacency_mask=self.graph_data.adjacency_mask
            )

            # Use attention from the last layer, averaged across heads
            last_layer_attn = attentions[-1]  # [num_edges, num_heads]
            avg_attention = last_layer_attn.mean(dim=-1).cpu().numpy()  # [num_edges]

            for i in range(edge_index.shape[1]):
                src_idx, tgt_idx = edge_index[0, i], edge_index[1, i]
                src_id = self.graph_data.idx_to_node[src_idx]
                tgt_id = self.graph_data.idx_to_node[tgt_idx]
                attention_dict[(src_id, tgt_id)] = float(avg_attention[i])

            logging.debug(f"Extracted attention scores for {len(attention_dict)} edges")

        except Exception as e:
            # Fallback to edge weights if attention extraction fails
            logging.warning(f"Attention extraction failed: {e}, using edge weights as fallback")
            edge_weights = self.graph_data.edge_weights.cpu().numpy()

            for i in range(edge_index.shape[1]):
                src_idx, tgt_idx = edge_index[0, i], edge_index[1, i]
                src_id = self.graph_data.idx_to_node[src_idx]
                tgt_id = self.graph_data.idx_to_node[tgt_idx]
                attention_dict[(src_id, tgt_id)] = float(edge_weights[i])

        return attention_dict

    def extract_and_rank_paths(self, query_emb: torch.Tensor,
                               start_nodes: Optional[List[str]] = None,
                               end_nodes: Optional[List[str]] = None) -> List[PathInfo]:
        """
        Extract and rank reasoning paths.

        Args:
            query_emb: Query embedding
            start_nodes: Starting nodes (if None, use top-k from query)
            end_nodes: Target nodes (if None, use top-k from PPR or query)

        Returns:
            List of ranked PathInfo objects
        """
        # Get start nodes if not provided
        if start_nodes is None:
            start_nodes, _ = self.get_top_k_nodes(query_emb, k=min(5, self.top_k_nodes // 4))
            logging.info(f"Using {len(start_nodes)} start nodes from query matching")

        # Get end nodes if not provided
        if end_nodes is None:
            # Use PPR to find relevant end nodes
            ppr_scores = self.propagate_ppr(query_emb)
            top_indices = np.argsort(ppr_scores)[-self.top_k_nodes:][::-1]
            end_nodes = [self.graph_data.idx_to_node[idx] for idx in top_indices]
            logging.info(f"Using {len(end_nodes)} end nodes from PPR")

        # Log diagnostic information
        logging.debug(f"Start nodes: {start_nodes[:3]}... (showing first 3)")
        logging.debug(f"End nodes: {end_nodes[:3]}... (showing first 3)")

        # Check overlap - this is normal when query matches relevant entities
        start_set = set(start_nodes)
        end_set = set(end_nodes)
        overlap = start_set.intersection(end_set)
        if overlap:
            logging.debug(f"Found {len(overlap)} nodes that are both start and end nodes (direct matches)")

        # Extract paths using BFS on real graph
        paths = extract_paths_bfs(
            G=self.graph_data.G,
            start_nodes=start_nodes,
            end_nodes=end_nodes,
            max_depth=self.max_path_length,
            max_paths=self.max_paths_per_query * 10  # Extract more, then rank
        )

        if not paths:
            logging.warning("No paths found between start and end nodes")
            logging.warning(f"  Start nodes: {len(start_nodes)}, End nodes: {len(end_nodes)}")
            logging.warning(f"  Max path length: {self.max_path_length}")
            logging.warning(f"  Graph has {self.graph_data.G.number_of_nodes()} nodes, {self.graph_data.G.number_of_edges()} edges")

            # Try to diagnose connectivity issue
            if start_nodes and end_nodes:
                sample_start = start_nodes[0]
                if sample_start in self.graph_data.G:
                    neighbors = list(self.graph_data.G.neighbors(sample_start))
                    logging.warning(f"  Sample start node '{sample_start}' has {len(neighbors)} outgoing edges")

            return []

        # Get attention scores for path ranking
        attention_scores = self.extract_attention_scores(query_emb)

        # Rank paths
        ranked_paths = rank_paths(
            paths=paths,
            G=self.graph_data.G,
            attention_scores=attention_scores,
            top_k=self.max_paths_per_query,
            alpha=0.5  # Weight between composite_importance and attention
        )

        # Filter by minimum score
        before_filter = len(ranked_paths)
        ranked_paths = [p for p in ranked_paths if p.score >= self.min_path_score]
        if before_filter > len(ranked_paths):
            logging.info(f"Filtered out {before_filter - len(ranked_paths)} paths with score < {self.min_path_score}")

        logging.info(f"Extracted and ranked {len(ranked_paths)} paths")

        return ranked_paths

    def reason(self, query_text: str, text_encoder=None,
              method: str = 'ppr') -> Dict[str, Any]:
        """
        Main reasoning function.

        Args:
            query_text: Natural language query
            text_encoder: Optional text encoder
            method: 'ppr' or 'gnn' for propagation method

        Returns:
            Reasoning results dictionary containing:
            - query: original query
            - top_nodes: top-k relevant nodes
            - paths: reasoning paths
            - explanations: formatted path explanations
        """
        logging.info("="*60)
        logging.info(f"Reasoning for query: {query_text}")
        logging.info("="*60)

        # 1. Encode query
        query_emb = self.encode_query(query_text, text_encoder)

        # 2. Get top-k nodes
        top_nodes, top_scores = self.get_top_k_nodes(query_emb)

        # 3. Propagate (optional, for end node selection)
        if method == 'ppr':
            ppr_scores = self.propagate_ppr(query_emb)
            top_ppr_indices = np.argsort(ppr_scores)[-self.top_k_nodes:][::-1]
            end_nodes = [self.graph_data.idx_to_node[idx] for idx in top_ppr_indices]
        else:
            end_nodes = top_nodes

        # 4. Extract and rank paths
        paths = self.extract_and_rank_paths(
            query_emb=query_emb,
            start_nodes=top_nodes[:5],  # Use top 5 as start
            end_nodes=end_nodes
        )

        # 5. Format explanations
        explanations = [
            format_path_explanation(path, self.graph_data.G, include_scores=True)
            for path in paths
        ]

        # 6. Compile results with chunk_id information for source tracing
        results = {
            'query': query_text,
            'top_nodes': [
                {
                    'id': node_id,
                    'name': self.graph_data.G.nodes[node_id].get('name', node_id) if node_id in self.graph_data.G.nodes else node_id,
                    'score': float(score),
                    # Add chunk_id for source reference
                    'chunk_ids': self._extract_chunk_ids(node_id)
                }
                for node_id, score in zip(top_nodes, top_scores)
            ],
            'paths': [
                {
                    'path': path.path,
                    'score': path.score,
                    'edge_types': path.edge_types,
                    'explanation': exp,
                    # Add chunk_ids from all nodes in path
                    'chunk_ids': self._extract_path_chunk_ids(path.path)
                }
                for path, exp in zip(paths, explanations)
            ],
            'num_paths': len(paths)
        }

        logging.info(f"Reasoning complete: {len(top_nodes)} top nodes, {len(paths)} paths")
        logging.info("="*60)

        return results

    def _extract_chunk_ids(self, node_id: str) -> List[str]:
        """
        Extract chunk_ids from a node.

        Args:
            node_id: Node ID

        Returns:
            List of chunk IDs associated with this node
        """
        if node_id not in self.graph_data.G.nodes:
            return []

        node_data = self.graph_data.G.nodes[node_id]
        chunk_ids = node_data.get('chunk_id') or node_data.get('text_unit_ids', [])

        if isinstance(chunk_ids, str):
            return [chunk_ids]
        elif isinstance(chunk_ids, list):
            return chunk_ids
        else:
            return []

    def _extract_path_chunk_ids(self, path: List[str]) -> List[str]:
        """
        Extract all chunk_ids from nodes in a path.

        Args:
            path: List of node IDs forming a path

        Returns:
            List of unique chunk IDs from all nodes in the path
        """
        all_chunks = set()
        for node_id in path:
            chunk_ids = self._extract_chunk_ids(node_id)
            all_chunks.update(chunk_ids)
        return list(all_chunks)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    print("✓ Inference module loaded successfully")

