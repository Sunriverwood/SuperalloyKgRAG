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
import hashlib
from typing import List, Dict, Tuple, Optional, Any, Callable
from openai import OpenAI

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
                 gnn_model: QueryAwareRGAT, query_matcher, device: str = 'cpu',
                 llm_client: Optional[OpenAI] = None, embedding_model: Optional[str] = None):
        """
        Args:
            config: Configuration dictionary
            graph_data: GraphData object
            gnn_model: Trained QueryAwareRGAT model
            query_matcher: Trained query-entity matcher
            device: 'cpu' or 'cuda'
            llm_client: OpenAI client for keyword extraction (optional)
            embedding_model: Embedding model name for keyword encoding (optional)
        """
        self.config = config
        self.reasoning_config = config.get('reasoning', {})
        self.inference_config = self.reasoning_config.get('inference', {})

        self.graph_data = graph_data
        self.device = torch.device(device)
        self.gnn = gnn_model.to(self.device)
        self.query_matcher = query_matcher.to(self.device)

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

        # Default reasoning method
        self.default_method = self.inference_config.get('default_method', 'gnn')

        # Use direct similarity for node scoring (bypasses GNN+QueryMatcher)
        # Set to True if model is not trained or results are poor
        self.use_direct_similarity = self.inference_config.get('use_direct_similarity', False)

        # Hybrid scoring weights (only used when use_direct_similarity=False)
        self.direct_similarity_weight = self.inference_config.get('direct_similarity_weight', 0.7)
        self.gnn_matcher_weight = self.inference_config.get('gnn_matcher_weight', 0.3)

        # Keyword enhancement configuration
        self.keyword_config = self.inference_config.get('keyword_enhancement', {})
        self.keyword_enhancement_enabled = self.keyword_config.get('enabled', True)
        self.max_keywords = self.keyword_config.get('max_keywords', 5)
        self.keyword_cache_enabled = self.keyword_config.get('cache_enabled', True)

        # PPR and GNN fusion strategies
        self.ppr_strategy = self.keyword_config.get('ppr_strategy', {
            'fusion_method': 'max_pool',
            'original_query_weight': 0.4,
            'keyword_weight': 0.6
        })
        self.gnn_strategy = self.keyword_config.get('gnn_strategy', {
            'fusion_method': 'weighted_avg',
            'original_query_weight': 0.7,
            'keyword_weight': 0.3
        })

        # LLM client for keyword extraction
        self.llm_client = llm_client
        self.embedding_model = embedding_model

        # Keyword cache (in-memory)
        self._keyword_cache: Dict[str, List[str]] = {}

        logging.info("GraphReasoner initialized")
        logging.info(f"  PPR alpha: {self.ppr_alpha}")
        logging.info(f"  Max path length: {self.max_path_length}")
        logging.info(f"  Device: {self.device}")
        logging.info(f"  Default method: {self.default_method}")
        logging.info(f"  Use direct similarity only: {self.use_direct_similarity}")
        if not self.use_direct_similarity:
            logging.info(f"  Hybrid scoring weights: direct={self.direct_similarity_weight}, gnn_matcher={self.gnn_matcher_weight}")
        logging.info(f"  Keyword enhancement: {'enabled' if self.keyword_enhancement_enabled else 'disabled'}")
        if self.keyword_enhancement_enabled:
            logging.info(f"    Max keywords: {self.max_keywords}")
            logging.info(f"    Cache enabled: {self.keyword_cache_enabled}")

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

    def _get_cache_key(self, query_text: str) -> str:
        """Generate cache key for keyword extraction."""
        return hashlib.md5(query_text.encode('utf-8')).hexdigest()

    def _extract_keywords(self, query_text: str) -> List[str]:
        """
        Extract keywords from query using LLM API.

        Args:
            query_text: Natural language query

        Returns:
            List of extracted keywords (entities, terms, concepts)
        """
        # Generate cache key
        cache_key = self._get_cache_key(query_text)

        # Check cache first
        if self.keyword_cache_enabled:
            if cache_key in self._keyword_cache:
                logging.debug(f"Keyword cache hit for query")
                return self._keyword_cache[cache_key]

        # Check if LLM client is available
        if self.llm_client is None:
            logging.warning("LLM client not available for keyword extraction")
            return []

        try:
            # Keyword extraction prompt
            prompt = f"""从以下问题中提取关键实体和术语，用于知识图谱检索。

问题：{query_text}

要求：
1. 提取材料名称（如合金名、牌号）
2. 提取性能指标（如强度、硬度、蠕变）
3. 提取工艺方法（如热处理、时效）
4. 提取化学元素和成分
5. 提取其他专业术语

输出格式：仅输出关键词列表，用逗号分隔，不要其他内容。
最多输出{self.max_keywords}个最重要的关键词。

示例输出：镍基高温合金,γ'相,时效处理,蠕变性能"""

            response = self.llm_client.chat.completions.create(
                model=self.config.get('query', {}).get('generation_model', 'qwen3-max'),
                messages=[
                    {"role": "system", "content": "你是一个专业的材料科学关键词提取专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )

            # Parse response
            keywords_text = response.choices[0].message.content.strip()
            # Handle /no_think format
            if "</think>" in keywords_text:
                keywords_text = keywords_text.split("</think>")[-1].strip()

            keywords = [kw.strip() for kw in keywords_text.split(',') if kw.strip()]
            keywords = keywords[:self.max_keywords]  # Limit to max_keywords

            logging.info(f"Extracted {len(keywords)} keywords: {keywords}")

            # Cache the result
            if self.keyword_cache_enabled:
                self._keyword_cache[cache_key] = keywords

            return keywords

        except Exception as e:
            logging.warning(f"Keyword extraction failed: {e}")
            return []

    def _encode_keywords(self, keywords: List[str], text_encoder: Callable) -> List[torch.Tensor]:
        """
        Encode multiple keywords to embedding vectors.

        Args:
            keywords: List of keyword strings
            text_encoder: Text encoding function

        Returns:
            List of keyword embedding tensors
        """
        embeddings = []
        for keyword in keywords:
            try:
                emb = text_encoder(keyword)
                emb_tensor = torch.tensor(emb, dtype=torch.float32, device=self.device)
                embeddings.append(emb_tensor)
            except Exception as e:
                logging.warning(f"Failed to encode keyword '{keyword}': {e}")
        return embeddings

    def _compute_enhanced_query_emb(self, original_emb: torch.Tensor,
                                     keyword_embs: List[torch.Tensor],
                                     method: str = 'gnn') -> torch.Tensor:
        """
        Compute enhanced query embedding by fusing original query with keywords.

        Args:
            original_emb: Original query embedding [embed_dim]
            keyword_embs: List of keyword embeddings
            method: 'ppr' or 'gnn' - determines fusion strategy

        Returns:
            Enhanced query embedding [embed_dim]
        """
        if not keyword_embs:
            logging.debug("No keyword embeddings, using original query embedding")
            return original_emb

        # Stack keyword embeddings
        keyword_stack = torch.stack(keyword_embs, dim=0)  # [num_keywords, embed_dim]

        # Select strategy based on method
        if method == 'ppr':
            strategy = self.ppr_strategy
        else:
            strategy = self.gnn_strategy

        fusion_method = strategy.get('fusion_method', 'weighted_avg')
        original_weight = strategy.get('original_query_weight', 0.5)
        keyword_weight = strategy.get('keyword_weight', 0.5)

        if fusion_method == 'max_pool':
            # Max pooling: take max across all keywords for each dimension
            # This expands coverage to capture diverse entities
            keyword_pooled = keyword_stack.max(dim=0).values  # [embed_dim]
            # Weighted fusion with original
            enhanced_emb = original_weight * original_emb + keyword_weight * keyword_pooled

        elif fusion_method == 'weighted_avg':
            # Weighted average: average of keywords, then fuse with original
            keyword_avg = keyword_stack.mean(dim=0)  # [embed_dim]
            enhanced_emb = original_weight * original_emb + keyword_weight * keyword_avg

        elif fusion_method == 'concat_project':
            # Concatenate and project (requires additional linear layer)
            # For now, fall back to weighted average
            keyword_avg = keyword_stack.mean(dim=0)
            enhanced_emb = original_weight * original_emb + keyword_weight * keyword_avg

        else:
            logging.warning(f"Unknown fusion method '{fusion_method}', using weighted average")
            keyword_avg = keyword_stack.mean(dim=0)
            enhanced_emb = original_weight * original_emb + keyword_weight * keyword_avg

        # Normalize the enhanced embedding
        enhanced_emb = F.normalize(enhanced_emb, p=2, dim=0)

        logging.debug(f"Enhanced query embedding using '{fusion_method}' strategy for {method}")
        return enhanced_emb

    def _get_tensor(self, name: str) -> torch.Tensor:
        """Helper to get a tensor from graph_data and ensure it's on self.device"""
        if not hasattr(self.graph_data, name):
            raise AttributeError(f"GraphData missing attribute: {name}")

        tensor = getattr(self.graph_data, name)
        if tensor is None:
            return None

        if isinstance(tensor, torch.Tensor):
            if tensor.device != self.device:
                return tensor.to(self.device)
        return tensor

    @torch.no_grad()
    def score_nodes(self, query_emb: torch.Tensor, use_gnn: bool = True,
                    use_direct_similarity: bool = False) -> torch.Tensor:
        """
        Score all nodes based on query relevance.

        Args:
            query_emb: Query embedding [embed_dim]
            use_gnn: If True, use query-aware GNN; else use GNN without query
            use_direct_similarity: If True, use direct cosine similarity instead of QueryMatcher
                                   This is useful when model is not well-trained

        Returns:
            Node scores [num_nodes]
        """
        # Debug: Log query embedding stats to verify it changes with different queries
        logging.info(f"[DEBUG] Query embedding: norm={query_emb.norm().item():.4f}, "
                    f"mean={query_emb.mean().item():.6f}, std={query_emb.std().item():.4f}, "
                    f"first_3={query_emb[:3].tolist()}")

        # Get original node embeddings
        x = self._get_tensor('node_embeddings')

        # Compute direct cosine similarity with original embeddings
        # This ensures scores change with different queries
        query_norm = F.normalize(query_emb.unsqueeze(0), p=2, dim=1)  # [1, embed_dim]
        nodes_norm = F.normalize(x, p=2, dim=1)  # [num_nodes, embed_dim]
        direct_scores = torch.mm(query_norm, nodes_norm.t()).squeeze(0)  # [num_nodes]

        logging.info(f"[DEBUG] Direct similarity scores: min={direct_scores.min().item():.4f}, "
                    f"max={direct_scores.max().item():.4f}, mean={direct_scores.mean().item():.4f}")

        # Option 1: Only use direct similarity
        if use_direct_similarity:
            return direct_scores

        # Option 2: Hybrid approach - combine direct similarity with GNN-enhanced scores
        # This ensures query relevance while also leveraging graph structure
        edge_index = self._get_tensor('edge_index')
        edge_weights = self._get_tensor('edge_weights')
        edge_types = self._get_tensor('edge_types')
        edge_type_embeddings_full = self._get_tensor('edge_type_embeddings')
        adjacency_mask = self._get_tensor('adjacency_mask')

        # Perform indexing on the correct device
        edge_type_emb = edge_type_embeddings_full[edge_types]

        # Get GNN-enhanced node embeddings
        node_embeddings = self.gnn(
            x=x,
            edge_index=edge_index,
            edge_type_emb=edge_type_emb,
            edge_weights=edge_weights,
            query_emb=query_emb if use_gnn else None,
            adjacency_mask=adjacency_mask
        )

        # Debug: Log GNN output stats
        logging.info(f"[DEBUG] GNN output: mean={node_embeddings.mean().item():.6f}, "
                    f"std={node_embeddings.std().item():.4f}")

        # Compute QueryMatcher scores
        matcher_scores = self.query_matcher(query_emb, node_embeddings)

        logging.info(f"[DEBUG] Matcher scores: min={matcher_scores.min().item():.4f}, "
                    f"max={matcher_scores.max().item():.4f}, mean={matcher_scores.mean().item():.4f}")

        # Hybrid scoring: combine direct similarity with matcher scores
        # Normalize both to similar scales before combining
        direct_scores_normalized = (direct_scores - direct_scores.mean()) / (direct_scores.std() + 1e-8)
        matcher_scores_normalized = (matcher_scores - matcher_scores.mean()) / (matcher_scores.std() + 1e-8)

        # Use configured weights
        # Higher direct_weight ensures the results change with different queries
        direct_weight = self.direct_similarity_weight
        matcher_weight = self.gnn_matcher_weight

        hybrid_scores = direct_weight * direct_scores_normalized + matcher_weight * matcher_scores_normalized

        logging.info(f"[DEBUG] Hybrid scores (direct={direct_weight}, matcher={matcher_weight}): "
                    f"min={hybrid_scores.min().item():.4f}, max={hybrid_scores.max().item():.4f}, "
                    f"mean={hybrid_scores.mean().item():.4f}")

        return hybrid_scores

    @torch.no_grad()
    def get_top_k_nodes(self, query_emb: torch.Tensor, k: Optional[int] = None,
                        use_direct_similarity: Optional[bool] = None) -> Tuple[List[str], torch.Tensor]:
        """
        Get top-k most relevant nodes for the query.

        Args:
            query_emb: Query embedding
            k: Number of top nodes to return (uses config if None)
            use_direct_similarity: If True, use direct cosine similarity (uses config if None)

        Returns:
            node_ids: List of top-k node IDs
            scores: Scores for top-k nodes
        """
        if k is None:
            k = self.top_k_nodes

        if use_direct_similarity is None:
            use_direct_similarity = self.use_direct_similarity

        # Score all nodes
        scores = self.score_nodes(query_emb, use_gnn=True, use_direct_similarity=use_direct_similarity)

        # Get top-k
        top_k_scores, top_k_indices = torch.topk(scores, k=min(k, len(scores)))

        # Convert indices to node IDs
        top_k_node_ids = [self.graph_data.idx_to_node[idx.item()] for idx in top_k_indices]

        logging.info(f"Top-{k} nodes retrieved, score range: [{top_k_scores.min():.4f}, {top_k_scores.max():.4f}]")

        return top_k_node_ids, top_k_scores

    def build_transition_matrix(self) -> 'scipy.sparse.csr_matrix':
        """
        Build transition matrix for PPR using composite_importance as edge weights.
        Uses sparse matrix to avoid memory issues with large graphs.

        Returns:
            Transition matrix P [num_nodes, num_nodes] as sparse CSR matrix
            P[i, j] = normalized weight of edge i -> j
        """
        from scipy.sparse import csr_matrix

        num_nodes = self.graph_data.num_nodes

        # Use edge weights (composite_importance)
        edge_index = self.graph_data.edge_index.cpu().numpy()
        edge_weights = self.graph_data.edge_weights.cpu().numpy()

        # Build sparse matrix using COO format, then convert to CSR
        row_indices = edge_index[0]  # source nodes
        col_indices = edge_index[1]  # target nodes

        P = csr_matrix((edge_weights, (row_indices, col_indices)), shape=(num_nodes, num_nodes))

        # Normalize rows (out-degree normalization)
        row_sums = np.array(P.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0  # Avoid division by zero

        # Create diagonal matrix for normalization
        inv_row_sums = 1.0 / row_sums
        D_inv = csr_matrix((inv_row_sums, (np.arange(num_nodes), np.arange(num_nodes))), shape=(num_nodes, num_nodes))

        # Normalized transition matrix: P_norm = D^{-1} * P
        P = D_inv @ P

        return P

    @torch.no_grad()
    def propagate_ppr(self, query_emb: torch.Tensor, alpha: Optional[float] = None,
                      max_iter: Optional[int] = None, tol: Optional[float] = None) -> np.ndarray:
        """
        Personalized PageRank propagation with query-aware initialization.
        Uses sparse matrix operations to handle large graphs efficiently.

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

        # Build sparse transition matrix
        P = self.build_transition_matrix()

        # PPR iteration with sparse matrix
        pi = pi_0.copy()

        for iteration in range(max_iter):
            # Sparse matrix multiplication: pi @ P returns a dense array
            # For row vector @ sparse matrix, use P.T @ pi.T then transpose
            pi_P = P.T.dot(pi)  # More efficient for sparse: (P^T * pi^T)^T = pi * P
            pi_new = alpha * pi_0 + (1 - alpha) * pi_P

            # Check convergence
            diff = np.abs(pi_new - pi).max()
            pi = pi_new

            if diff < tol:
                logging.info(f"PPR converged at iteration {iteration + 1}, diff={diff:.2e}")
                break
        else:
            logging.warning(f"PPR did not converge after {max_iter} iterations")

        return pi

    def _get_edge_weight_scores(self) -> Dict[Tuple[str, str], float]:
        """
        Get edge weight scores for path ranking (without using GNN attention).

        Returns:
            Dictionary mapping (src_id, tgt_id) to edge weight score
        """
        edge_index_cpu = self.graph_data.edge_index.cpu().numpy()
        edge_weights_cpu = self.graph_data.edge_weights.cpu().numpy()

        scores = {}
        for i in range(edge_index_cpu.shape[1]):
            src_idx, tgt_idx = edge_index_cpu[0, i], edge_index_cpu[1, i]
            src_id = self.graph_data.idx_to_node[src_idx]
            tgt_id = self.graph_data.idx_to_node[tgt_idx]
            scores[(src_id, tgt_id)] = float(edge_weights_cpu[i])

        logging.debug(f"Created edge weight scores for {len(scores)} edges")
        return scores

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

        # Prepare inputs on correct device
        x = self._get_tensor('node_embeddings')
        edge_index_tensor = self._get_tensor('edge_index')
        edge_weights = self._get_tensor('edge_weights')
        edge_types = self._get_tensor('edge_types')
        edge_type_embeddings_full = self._get_tensor('edge_type_embeddings')
        adjacency_mask = self._get_tensor('adjacency_mask')

        edge_type_emb = edge_type_embeddings_full[edge_types]

        # For mapping back to node IDs, we need edge_index on CPU
        edge_index_cpu = self.graph_data.edge_index.cpu().numpy()

        # Use GNN's extract_attention method to get real attention weights
        try:
            _, attentions = self.gnn.extract_attention(
                x=x,
                edge_index=edge_index_tensor,
                edge_type_emb=edge_type_emb,
                edge_weights=edge_weights,
                query_emb=query_emb,
                adjacency_mask=adjacency_mask
            )

            # Use attention from the last layer, averaged across heads
            last_layer_attn = attentions[-1]  # [num_edges, num_heads]
            avg_attention = last_layer_attn.mean(dim=-1).cpu().numpy()  # [num_edges]

            for i in range(edge_index_cpu.shape[1]):
                src_idx, tgt_idx = edge_index_cpu[0, i], edge_index_cpu[1, i]
                src_id = self.graph_data.idx_to_node[src_idx]
                tgt_id = self.graph_data.idx_to_node[tgt_idx]
                attention_dict[(src_id, tgt_id)] = float(avg_attention[i])

            logging.debug(f"Extracted attention scores for {len(attention_dict)} edges")

        except Exception as e:
            # Fallback to edge weights if attention extraction fails
            logging.warning(f"Attention extraction failed: {e}, using edge weights as fallback")
            edge_weights_cpu = self.graph_data.edge_weights.cpu().numpy()

            for i in range(edge_index_cpu.shape[1]):
                src_idx, tgt_idx = edge_index_cpu[0, i], edge_index_cpu[1, i]
                src_id = self.graph_data.idx_to_node[src_idx]
                tgt_id = self.graph_data.idx_to_node[tgt_idx]
                attention_dict[(src_id, tgt_id)] = float(edge_weights_cpu[i])

        return attention_dict

    def extract_and_rank_paths(self, query_emb: torch.Tensor,
                               start_nodes: Optional[List[str]] = None,
                               end_nodes: Optional[List[str]] = None,
                               use_attention: bool = True) -> List[PathInfo]:
        """
        Extract and rank reasoning paths.

        Args:
            query_emb: Query embedding
            start_nodes: Starting nodes (if None, use top-k from query)
            end_nodes: Target nodes (if None, use top-k from PPR or query)
            use_attention: If True, use GNN attention scores; else use edge weights only

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
            logging.warning(
                f"  Graph has {self.graph_data.G.number_of_nodes()} nodes, {self.graph_data.G.number_of_edges()} edges")

            # Try to diagnose connectivity issue
            if start_nodes and end_nodes:
                sample_start = start_nodes[0]
                if sample_start in self.graph_data.G:
                    neighbors = list(self.graph_data.G.neighbors(sample_start))
                    logging.warning(f"  Sample start node '{sample_start}' has {len(neighbors)} outgoing edges")

            return []

        # Get scores for path ranking
        # If use_attention=False or use_direct_similarity=True, use edge weights only
        if use_attention and not self.use_direct_similarity:
            attention_scores = self.extract_attention_scores(query_emb)
        else:
            # Use edge weights as scores (no GNN attention needed)
            logging.info("Using edge weights for path ranking (skipping GNN attention)")
            attention_scores = self._get_edge_weight_scores()

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
               method: str = None) -> Dict[str, Any]:
        """
        Main reasoning function with keyword enhancement.

        Args:
            query_text: Natural language query
            text_encoder: Optional text encoder
            method: 'ppr' or 'gnn' for propagation method (uses default if None)

        Returns:
            Reasoning results dictionary containing:
            - query: original query
            - top_nodes: top-k relevant nodes
            - paths: reasoning paths
            - explanations: formatted path explanations
            - keywords: extracted keywords (if enhancement enabled)
        """
        # Use default method if not specified
        if method is None:
            method = self.default_method

        logging.info("=" * 60)
        logging.info(f"Reasoning for query: {query_text}")
        logging.info(f"Method: {method}, Keyword enhancement: {self.keyword_enhancement_enabled}")
        logging.info("=" * 60)

        # 1. Encode original query
        original_query_emb = self.encode_query(query_text, text_encoder)

        # Debug: Check if query embedding varies with different queries
        logging.info(f"Original query embedding: norm={original_query_emb.norm():.4f}, mean={original_query_emb.mean():.4f}, first_5={original_query_emb[:5].tolist()}")

        # 2. Keyword enhancement (if enabled and text_encoder available)
        keywords = []
        if self.keyword_enhancement_enabled and text_encoder is not None:
            # Extract keywords using LLM
            keywords = self._extract_keywords(query_text)

            if keywords:
                # Encode keywords
                keyword_embs = self._encode_keywords(keywords, text_encoder)

                # Compute enhanced query embedding based on method
                query_emb = self._compute_enhanced_query_emb(
                    original_emb=original_query_emb,
                    keyword_embs=keyword_embs,
                    method=method
                )
                logging.info(f"Query embedding enhanced with {len(keywords)} keywords")
            else:
                query_emb = original_query_emb
                logging.info("No keywords extracted, using original query embedding")
        else:
            query_emb = original_query_emb
            if not self.keyword_enhancement_enabled:
                logging.debug("Keyword enhancement disabled")
            elif text_encoder is None:
                logging.debug("No text encoder provided for keyword encoding")

        # 3. Get top-k nodes using enhanced query
        logging.info(f"[DEBUG] Before get_top_k_nodes: query_emb norm={query_emb.norm().item():.4f}, device={query_emb.device}")
        top_nodes, top_scores = self.get_top_k_nodes(query_emb)

        # 4. Propagate (optional, for end node selection)
        if method == 'ppr':
            ppr_scores = self.propagate_ppr(query_emb)
            top_ppr_indices = np.argsort(ppr_scores)[-self.top_k_nodes:][::-1]
            end_nodes = [self.graph_data.idx_to_node[idx] for idx in top_ppr_indices]
        else:
            end_nodes = top_nodes

        # 5. Extract and rank paths
        # When using direct similarity, don't use GNN attention for path ranking
        paths = self.extract_and_rank_paths(
            query_emb=query_emb,
            start_nodes=top_nodes[:5],  # Use top 5 as start
            end_nodes=end_nodes,
            use_attention=not self.use_direct_similarity
        )

        # 6. Format explanations
        explanations = [
            format_path_explanation(path, self.graph_data.G, include_scores=True)
            for path in paths
        ]

        # 7. Compile results with chunk_id information for source tracing
        results = {
            'query': query_text,
            'method': method,
            'keywords': keywords,  # Include extracted keywords
            'top_nodes': [
                {
                    'id': node_id,
                    'name': self.graph_data.G.nodes[node_id].get('name',
                                                                 node_id) if node_id in self.graph_data.G.nodes else node_id,
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
        logging.info("=" * 60)

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