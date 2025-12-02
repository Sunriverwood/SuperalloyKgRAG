"""
Query-Aware Relational Graph Attention Network (RGAT)

Implements a GNN encoder that:
1. Performs message passing with relation-specific attention
2. Incorporates query vectors for query-aware reasoning
3. Respects graph constraints via adjacency masking
4. Uses edge weights (composite_importance) as priors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging


class RGATLayer(nn.Module):
    """
    Single layer of Relational Graph Attention Network with query awareness.

    Computes attention scores: e_uv = φ(h_u, h_v, r_uv, q)
    where:
    - h_u, h_v: node embeddings
    - r_uv: edge type embedding
    - q: query embedding (optional)
    """

    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4,
                 dropout: float = 0.1, use_edge_weights: bool = True,
                 query_dim: Optional[int] = None, edge_type_dim: Optional[int] = None):
        """
        Args:
            in_dim: Input node feature dimension
            out_dim: Output node feature dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            use_edge_weights: Whether to use edge weights as attention priors
            query_dim: Query embedding dimension (if None, assume same as in_dim)
            edge_type_dim: Edge type embedding dimension (if None, assume same as in_dim)
        """
        super(RGATLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.use_edge_weights = use_edge_weights
        self.query_dim = query_dim if query_dim is not None else in_dim
        self.edge_type_dim = edge_type_dim if edge_type_dim is not None else in_dim

        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"

        # Linear transformations for nodes
        self.W_src = nn.Linear(in_dim, out_dim, bias=False)
        self.W_dst = nn.Linear(in_dim, out_dim, bias=False)
        self.W_rel = nn.Linear(self.edge_type_dim, out_dim, bias=False)  # For edge type embeddings

        # Query transformation (if using query-aware attention)
        self.W_query = nn.Linear(self.query_dim, out_dim, bias=False)

        # Attention mechanism
        # e_uv = LeakyReLU(a^T [W_src h_u || W_dst h_v || W_rel r_uv || W_query q])
        self.attn = nn.Parameter(torch.randn(1, num_heads, 4 * self.head_dim))

        # Edge weight scaling (learnable)
        if use_edge_weights:
            self.edge_weight_scale = nn.Parameter(torch.ones(1))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.W_src.weight)
        nn.init.xavier_uniform_(self.W_dst.weight)
        nn.init.xavier_uniform_(self.W_rel.weight)
        nn.init.xavier_uniform_(self.W_query.weight)
        # For attention parameter, reshape to 2D, initialize, then reshape back
        # attn is [1, num_heads, 4*head_dim]
        attn_2d = self.attn.view(-1, 4 * self.head_dim)  # [num_heads, 4*head_dim]
        nn.init.xavier_uniform_(attn_2d)
        self.attn.data = attn_2d.view(1, self.num_heads, 4 * self.head_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_type_emb: torch.Tensor, edge_weights: Optional[torch.Tensor] = None,
                query_emb: Optional[torch.Tensor] = None,
                adjacency_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of RGAT layer.

        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_type_emb: Edge type embeddings for each edge [num_edges, embed_dim]
                          (already indexed from edge type embedding matrix)
            edge_weights: Edge weights (composite_importance) [num_edges]
            query_emb: Query embedding [query_dim] or [batch, query_dim]
            adjacency_mask: Binary adjacency matrix [num_nodes, num_nodes] for constraints

        Returns:
            Updated node features [num_nodes, out_dim]
        """
        num_nodes = x.size(0)

        # Linear transformations
        h_src = self.W_src(x)  # [num_nodes, out_dim]
        h_dst = self.W_dst(x)  # [num_nodes, out_dim]

        # Transform edge type embeddings (they come in original embedding dimension)
        h_rel = self.W_rel(edge_type_emb)  # [num_edges, out_dim]

        # Prepare query embedding (broadcast to all edges)
        if query_emb is not None:
            if query_emb.dim() == 1:
                query_emb = query_emb.unsqueeze(0)  # [1, query_dim]
            h_query = self.W_query(query_emb)  # [1, out_dim] or [batch, out_dim]

            # Broadcast to all edges
            if h_query.size(0) == 1:
                h_query = h_query.expand(edge_index.size(1), -1)  # [num_edges, out_dim]
        else:
            # No query - use zero vector
            h_query = torch.zeros(edge_index.size(1), self.out_dim, device=x.device)

        # Reshape for multi-head attention
        h_src = h_src.view(-1, self.num_heads, self.head_dim)  # [num_nodes, num_heads, head_dim]
        h_dst = h_dst.view(-1, self.num_heads, self.head_dim)
        h_rel = h_rel.view(-1, self.num_heads, self.head_dim)  # [num_edges, num_heads, head_dim]
        h_query = h_query.view(-1, self.num_heads, self.head_dim)  # [num_edges, num_heads, head_dim]

        # Compute attention scores
        src_idx = edge_index[0]  # [num_edges]
        dst_idx = edge_index[1]  # [num_edges]

        # Concatenate features for each edge
        # [num_edges, num_heads, 4 * head_dim]
        attn_input = torch.cat([
            h_src[src_idx],  # Source node
            h_dst[dst_idx],  # Target node
            h_rel,           # Edge type
            h_query          # Query
        ], dim=-1)

        # Compute attention coefficients
        # e_uv = a^T * concat_features
        e = (attn_input * self.attn).sum(dim=-1)  # [num_edges, num_heads]
        e = self.leaky_relu(e)

        # Incorporate edge weights as priors if available
        if self.use_edge_weights and edge_weights is not None:
            # Scale edge weights and add to attention logits
            edge_weight_contribution = edge_weights.unsqueeze(-1) * self.edge_weight_scale
            e = e + edge_weight_contribution  # [num_edges, num_heads]

        # Apply graph constraints via adjacency mask
        if adjacency_mask is not None:
            # Create mask for edges
            edge_mask = adjacency_mask[src_idx, dst_idx]  # [num_edges]
            edge_mask = edge_mask.unsqueeze(-1)  # [num_edges, 1]

            # Set attention to -inf for non-existing edges
            e = torch.where(edge_mask.bool(), e, torch.full_like(e, -1e9))

        # Softmax normalization per destination node
        # For each destination node, normalize over all incoming edges
        alpha = self.softmax_per_node(e, dst_idx, num_nodes)  # [num_edges, num_heads]
        alpha = self.dropout(alpha)

        # Message passing
        # Aggregate messages from neighbors
        h_dst_transformed = h_dst.view(-1, self.num_heads, self.head_dim)

        # Initialize output
        out = torch.zeros(num_nodes, self.num_heads, self.head_dim, device=x.device)

        # Aggregate: for each destination node, sum weighted source features
        src_features = h_src[src_idx]  # [num_edges, num_heads, head_dim]
        weighted_features = src_features * alpha.unsqueeze(-1)  # [num_edges, num_heads, head_dim]

        # Scatter add to destination nodes
        for i in range(self.num_heads):
            out[:, i, :].scatter_add_(
                0,
                dst_idx.unsqueeze(-1).expand(-1, self.head_dim),
                weighted_features[:, i, :]
            )

        # Concatenate heads and return
        out = out.view(num_nodes, -1)  # [num_nodes, out_dim]

        return out

    def softmax_per_node(self, scores: torch.Tensor, node_idx: torch.Tensor,
                         num_nodes: int) -> torch.Tensor:
        """
        Apply softmax normalization per node (over incoming edges).

        Args:
            scores: Attention scores [num_edges, num_heads]
            node_idx: Destination node indices [num_edges]
            num_nodes: Total number of nodes

        Returns:
            Normalized attention weights [num_edges, num_heads]
        """
        # Compute max per node for numerical stability
        max_scores = torch.full((num_nodes, scores.size(1)), -1e9, device=scores.device)
        max_scores.scatter_reduce_(0, node_idx.unsqueeze(-1).expand_as(scores),
                                   scores, reduce='amax', include_self=False)

        # Subtract max and compute exp
        scores_shifted = scores - max_scores[node_idx]
        exp_scores = torch.exp(scores_shifted)

        # Sum exp per node
        sum_exp = torch.zeros(num_nodes, scores.size(1), device=scores.device)
        sum_exp.scatter_add_(0, node_idx.unsqueeze(-1).expand_as(exp_scores), exp_scores)

        # Normalize
        alpha = exp_scores / (sum_exp[node_idx] + 1e-16)

        return alpha


class QueryAwareRGAT(nn.Module):
    """
    Multi-layer Query-Aware RGAT model for graph reasoning.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 3, num_heads: int = 4, dropout: float = 0.1,
                 use_edge_weights: bool = True, query_dim: Optional[int] = None,
                 edge_type_dim: Optional[int] = None):
        """
        Args:
            input_dim: Input node feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            num_layers: Number of RGAT layers
            num_heads: Number of attention heads per layer
            dropout: Dropout probability
            use_edge_weights: Whether to use edge weights
            query_dim: Query embedding dimension
            edge_type_dim: Edge type embedding dimension (constant across layers)
        """
        super(QueryAwareRGAT, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.edge_type_dim = edge_type_dim if edge_type_dim is not None else input_dim

        # Build layers
        self.layers = nn.ModuleList()

        # First layer: input_dim -> hidden_dim
        self.layers.append(
            RGATLayer(
                in_dim=input_dim,
                out_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_edge_weights=use_edge_weights,
                query_dim=query_dim,
                edge_type_dim=self.edge_type_dim
            )
        )

        # Middle layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 2):
            self.layers.append(
                RGATLayer(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    use_edge_weights=use_edge_weights,
                    query_dim=query_dim,
                    edge_type_dim=self.edge_type_dim
                )
            )

        # Last layer: hidden_dim -> output_dim
        if num_layers > 1:
            self.layers.append(
                RGATLayer(
                    in_dim=hidden_dim,
                    out_dim=output_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    use_edge_weights=use_edge_weights,
                    query_dim=query_dim,
                    edge_type_dim=self.edge_type_dim
                )
            )

        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim if i < num_layers - 1 else output_dim)
            for i in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_type_emb: torch.Tensor, edge_weights: Optional[torch.Tensor] = None,
                query_emb: Optional[torch.Tensor] = None,
                adjacency_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through all RGAT layers.

        Args:
            x: Initial node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_type_emb: Edge type embeddings [num_edges, input_dim]
            edge_weights: Edge weights [num_edges]
            query_emb: Query embedding [query_dim]
            adjacency_mask: Adjacency mask [num_nodes, num_nodes]

        Returns:
            Final node representations [num_nodes, output_dim]
        """
        h = x

        for i, layer in enumerate(self.layers):
            h_new = layer(h, edge_index, edge_type_emb, edge_weights,
                         query_emb, adjacency_mask)

            # Batch normalization
            h_new = self.batch_norms[i](h_new)

            # Residual connection (if dimensions match)
            if h.size(-1) == h_new.size(-1):
                h = h + h_new
            else:
                h = h_new

            # Activation and dropout (except last layer)
            if i < len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)

        return h


if __name__ == "__main__":
    # Test the RGAT model
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Create dummy data
    num_nodes = 100
    num_edges = 500
    input_dim = 768
    hidden_dim = 256
    output_dim = 256
    query_dim = 768

    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_type_emb = torch.randn(num_edges, input_dim)
    edge_weights = torch.rand(num_edges)
    query_emb = torch.randn(query_dim)

    # Create adjacency mask
    adjacency_mask = torch.zeros(num_nodes, num_nodes)
    adjacency_mask[edge_index[0], edge_index[1]] = 1.0

    # Create model
    model = QueryAwareRGAT(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=3,
        num_heads=4,
        dropout=0.1,
        use_edge_weights=True,
        query_dim=query_dim
    )

    # Forward pass
    output = model(x, edge_index, edge_type_emb, edge_weights, query_emb, adjacency_mask)

    print(f"✓ RGAT model test successful!")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
