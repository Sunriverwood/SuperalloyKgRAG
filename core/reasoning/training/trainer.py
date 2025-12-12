"""
Self-Supervised Training for Graph Reasoning

Implements three types of self-supervised tasks:
1. Link Prediction (edge reconstruction)
2. Graph Contrastive Learning (InfoNCE)
3. Pseudo Query-Entity Matching
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import json
import time

from core.reasoning.models.rgat import QueryAwareRGAT
from core.reasoning.data_loader import GraphData
from utils.graph_reasoning_utils import PseudoQueryGenerator

# --- 项目根目录定义 ---
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class LinkPredictionDecoder(nn.Module):
    """
    Decoder for link prediction task.
    Predicts whether an edge exists between two nodes.
    """

    def __init__(self, hidden_dim: int):
        super(LinkPredictionDecoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, h_u: torch.Tensor, h_v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_u: Source node embeddings [batch, hidden_dim]
            h_v: Target node embeddings [batch, hidden_dim]

        Returns:
            Edge existence scores [batch, 1]
        """
        h_concat = torch.cat([h_u, h_v], dim=-1)
        return self.mlp(h_concat)


class QueryEntityMatcher(nn.Module):
    """
    Matching function for query-entity similarity.
    f(q, v) → similarity score
    """

    def __init__(self, query_dim: int, entity_dim: int, hidden_dim: int = 256):
        super(QueryEntityMatcher, self).__init__()

        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.entity_proj = nn.Linear(entity_dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, query_emb: torch.Tensor, entity_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_emb: Query embedding [batch, query_dim] or [query_dim]
            entity_emb: Entity embeddings [num_entities, entity_dim] or [batch, entity_dim]

        Returns:
            Similarity scores
        """
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)  # [1, query_dim]

        q_proj = self.query_proj(query_emb)  # [batch, hidden_dim]
        e_proj = self.entity_proj(entity_emb)  # [num_entities, hidden_dim]

        # Compute pairwise similarity
        if q_proj.size(0) == 1 and e_proj.size(0) > 1:
            # Broadcasting: [1, hidden_dim] x [num_entities, hidden_dim]
            q_proj = q_proj.expand(e_proj.size(0), -1)

        h_concat = torch.cat([q_proj, e_proj], dim=-1)
        scores = self.mlp(h_concat).squeeze(-1)

        return scores


class GraphReasoningTrainer:
    """
    Trainer for self-supervised graph reasoning model.

    Combines three loss objectives:
    1. Link prediction loss
    2. Contrastive learning loss (optional)
    3. Pseudo query-entity matching loss
    """

    def __init__(self, config: Dict[str, Any], graph_data: GraphData, device: Union[str, torch.device] = 'cuda'):
        """
        Args:
            config: Configuration dictionary
            graph_data: GraphData object
            device: 'cpu' or 'cuda'
        """
        self.config = config
        self.reasoning_config = config.get('reasoning', {})
        self.model_config = self.reasoning_config.get('model', {})
        self.training_config = self.reasoning_config.get('training', {})

        self.graph_data = graph_data
        self.device = torch.device(device)

        # MEMORY OPTIMIZATION: Store which tensors are on which device
        logging.info(f"🔧 [Trainer Init] Graph data devices:")
        logging.info(f"  - node_embeddings: {graph_data.node_embeddings.device}")
        logging.info(f"  - edge_index: {graph_data.edge_index.device}")
        logging.info(f"  - edge_type_embeddings: {graph_data.edge_type_embeddings.device}")
        logging.info(f"  - Training device: {self.device}")

        # Model components
        self.gnn = self._build_gnn().to(self.device)
        self.link_decoder = LinkPredictionDecoder(self.model_config.get('hidden_dim', 256)).to(self.device)
        self.query_matcher = QueryEntityMatcher(
            query_dim=graph_data.embed_dim,
            entity_dim=self.model_config.get('hidden_dim', 256),
            hidden_dim=self.model_config.get('hidden_dim', 256)
        ).to(self.device)

        # Optimizer
        all_params = list(self.gnn.parameters()) + list(self.link_decoder.parameters()) + \
                     list(self.query_matcher.parameters())
        self.optimizer = torch.optim.Adam(
            all_params,
            lr=self.training_config.get('learning_rate', 0.001),
            weight_decay=self.training_config.get('weight_decay', 0.0001)
        )

        # Loss weights
        self.loss_weights = self.training_config.get('loss_weights', {
            'link_prediction': 1.0,
            'contrastive': 0.5,
            'pseudo_query': 1.0
        })

        # Training state
        self.epoch = 0
        self.best_loss = float('inf')
        self.patience_counter = 0

        # Pseudo query generator
        self.pseudo_query_gen = PseudoQueryGenerator(graph_data.G)

        logging.info("GraphReasoningTrainer initialized")
        logging.info(f"  GNN parameters: {sum(p.numel() for p in self.gnn.parameters()):,}")
        logging.info(f"  Device: {self.device}")

    def _build_gnn(self) -> QueryAwareRGAT:
        """Build the GNN model"""
        return QueryAwareRGAT(
            input_dim=self.graph_data.embed_dim,
            hidden_dim=self.model_config.get('hidden_dim', 256),
            output_dim=self.model_config.get('hidden_dim', 256),
            num_layers=self.model_config.get('num_layers', 3),
            num_heads=self.model_config.get('num_heads', 4),
            dropout=self.model_config.get('dropout', 0.1),
            use_edge_weights=self.model_config.get('use_edge_weights', True),
            query_dim=self.graph_data.embed_dim,
            edge_type_dim=self.graph_data.embed_dim  # Edge type embeddings stay in original dimension
        )

    def sample_negative_edges(self, num_samples: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Sample negative edges (non-existing edges).

        Args:
            num_samples: Number of negative samples

        Returns:
            Negative edge indices [2, num_samples]
        """
        num_nodes = self.graph_data.num_nodes
        negative_edges = []

        # Convert adjacency mask to set for fast lookup
        existing_edges = set()
        edge_index = self.graph_data.edge_index.cpu().numpy()
        for i in range(edge_index.shape[1]):
            existing_edges.add((edge_index[0, i], edge_index[1, i]))

        while len(negative_edges) < num_samples:
            u = np.random.randint(0, num_nodes)
            v = np.random.randint(0, num_nodes)

            if u != v and (u, v) not in existing_edges:
                negative_edges.append([u, v])

        dev = device or self.device
        return torch.tensor(negative_edges, dtype=torch.long, device=dev).T

    def link_prediction_loss(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute link prediction loss.

        Args:
            node_embeddings: Node representations from GNN [num_nodes, hidden_dim]

        Returns:
            Binary cross-entropy loss
        """
        dev = node_embeddings.device
        non_blocking = dev.type == 'cuda'

        # Positive samples (existing edges)
        pos_edge_index = self.graph_data.edge_index
        if pos_edge_index.device != dev:
            pos_edge_index = pos_edge_index.to(dev, non_blocking=non_blocking)

        pos_edge_weights = self.graph_data.edge_weights
        if pos_edge_weights.device != dev:
            pos_edge_weights = pos_edge_weights.to(dev, non_blocking=non_blocking)

        # Sample negative edges
        num_neg = min(pos_edge_index.size(1), self.training_config.get('batch_size', 512))
        neg_edge_index = self.sample_negative_edges(num_neg, device=dev)

        # Get embeddings for source and target nodes
        pos_src_emb = node_embeddings[pos_edge_index[0]]
        pos_tgt_emb = node_embeddings[pos_edge_index[1]]

        neg_src_emb = node_embeddings[neg_edge_index[0]]
        neg_tgt_emb = node_embeddings[neg_edge_index[1]]

        # Predict edge existence
        pos_scores = self.link_decoder(pos_src_emb, pos_tgt_emb).squeeze()
        neg_scores = self.link_decoder(neg_src_emb, neg_tgt_emb).squeeze()

        # Binary cross-entropy loss with edge weights
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores, torch.ones_like(pos_scores), weight=pos_edge_weights, reduction='mean'
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores, torch.zeros_like(neg_scores), reduction='mean'
        )

        return pos_loss + neg_loss

    def contrastive_loss(self, node_embeddings: torch.Tensor) -> torch.Tensor:
            """
            Compute contrastive learning loss (InfoNCE / NT-Xent).

            NOTE:
            The naive implementation builds an [N, N] similarity matrix, which will OOM for large graphs.
            This implementation supports:
            - Subsampling nodes (default) via training.contrastive_num_samples
            - Chunked logits computation via training.contrastive_chunk_size (no full [N, N] allocation)

            Args:
                node_embeddings: Node representations [num_nodes, hidden_dim]

            Returns:
                InfoNCE loss (scalar)
            """
            dev = node_embeddings.device
            N = node_embeddings.size(0)

            # Hyperparams (overridable in config)
            temp = float(self.training_config.get('contrastive_temperature', 0.2))
            noise_std = float(self.training_config.get('contrastive_noise_std', 0.1))

            # 1) Optional subsampling to cap the quadratic cost
            max_samples = self.training_config.get('contrastive_num_samples', 4096)
            if max_samples is not None:
                max_samples = int(max_samples)
                if N > max_samples:
                    idx = torch.randperm(N, device=dev)[:max_samples]
                    node_embeddings = node_embeddings.index_select(0, idx)
                    N = node_embeddings.size(0)

            # 2) Two augmented views
            aug1 = node_embeddings + torch.randn_like(node_embeddings) * noise_std
            aug2 = node_embeddings + torch.randn_like(node_embeddings) * noise_std

            # Normalize
            aug1 = F.normalize(aug1, dim=-1)
            aug2 = F.normalize(aug2, dim=-1)

            # 3) Chunked InfoNCE to avoid allocating full [N, N]
            chunk_size = self.training_config.get('contrastive_chunk_size', None)
            if chunk_size is not None:
                chunk_size = int(chunk_size)
                if chunk_size <= 0:
                    chunk_size = None

            if chunk_size is not None and N > chunk_size:
                aug2_t = aug2.T  # [hidden_dim, N]
                total = 0.0
                count = 0

                for s in range(0, N, chunk_size):
                    e = min(s + chunk_size, N)
                    B = e - s

                    logits = torch.mm(aug1[s:e], aug2_t) / temp  # [B, N]
                    lse = torch.logsumexp(logits, dim=1)  # [B]
                    diag_idx = torch.arange(s, e, device=dev)
                    diag = logits[torch.arange(B, device=dev), diag_idx]  # [B]

                    total = total + (-diag + lse).sum()
                    count += B

                    del logits, lse, diag

                return total / max(count, 1)

            # Fallback (safe when N is small)
            sim_matrix = torch.mm(aug1, aug2.T) / temp
            labels = torch.arange(N, device=dev)
            return F.cross_entropy(sim_matrix, labels)
    def pseudo_query_loss(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute pseudo query-entity matching loss.

        For each triplet (h, r, t):
        - Create pseudo query from (h, r)
        - Positive: tail entity t
        - Negatives: random sampled entities

        Args:
            node_embeddings: Node representations [num_nodes, hidden_dim]

        Returns:
            Ranking loss (BPR or cross-entropy)
        """
        dev = node_embeddings.device
        non_blocking = dev.type == 'cuda'

        # Generate triplets
        triplets = self.pseudo_query_gen.generate_triplets(
            max_triplets=self.training_config.get('batch_size', 512)
        )
        if not triplets:
            return torch.tensor(0.0, device=dev)

        losses = []
        num_negatives = self.training_config.get('num_negatives', 5)

        for head_id, relation, tail_id in triplets:
            if head_id not in self.graph_data.node_to_idx or tail_id not in self.graph_data.node_to_idx:
                continue

            head_idx = self.graph_data.node_to_idx[head_id]
            tail_idx = self.graph_data.node_to_idx[tail_id]

            # Pseudo query embedding (head embedding for now)
            query_emb = self.graph_data.node_embeddings[head_idx]
            if query_emb.device != dev:
                query_emb = query_emb.to(dev, non_blocking=non_blocking)

            # Positive sample
            pos_emb = node_embeddings[tail_idx].unsqueeze(0)  # [1, hidden_dim]

            # Negative samples
            neg_indices = torch.randint(0, self.graph_data.num_nodes, (num_negatives,), device=dev)
            neg_emb = node_embeddings[neg_indices]  # [num_negatives, hidden_dim]

            # Scores
            pos_score = self.query_matcher(query_emb, pos_emb)
            neg_scores = self.query_matcher(query_emb, neg_emb)

            # BPR loss
            for neg_score in neg_scores:
                losses.append(-F.logsigmoid(pos_score - neg_score))

        if not losses:
            return torch.tensor(0.0, device=dev)

        return torch.stack(losses).mean()

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of loss values
        """
        self.gnn.train()
        self.link_decoder.train()
        self.query_matcher.train()

        # Get gradient accumulation steps from config (default: 1)
        gradient_accumulation_steps = self.training_config.get('gradient_accumulation_steps', 1)

        # MEMORY OPTIMIZATION: Move ALL necessary tensors to device for forward pass
        # This ensures all tensors are on the same device and prevents OOM by keeping them on CPU when not needed

        logging.debug(f"🔧 Preparing tensors for forward pass on {self.device}...")

        # Move node embeddings
        node_embeddings_input = self.graph_data.node_embeddings
        if node_embeddings_input.device != self.device:
            node_embeddings_input = node_embeddings_input.to(self.device)

        # Move edge_index
        edge_index = self.graph_data.edge_index
        if edge_index.device != self.device:
            edge_index = edge_index.to(self.device)

        # Move edge_types
        edge_types = self.graph_data.edge_types
        if edge_types.device != self.device:
            edge_types = edge_types.to(self.device)

        # Move edge_weights
        edge_weights = self.graph_data.edge_weights
        if edge_weights.device != self.device:
            edge_weights = edge_weights.to(self.device)

        # CRITICAL FIX: Move edge_type_embeddings to device BEFORE indexing
        # Otherwise we get "indices should be on same device" error
        edge_type_embeddings_full = self.graph_data.edge_type_embeddings
        if edge_type_embeddings_full.device != self.device:
            edge_type_embeddings_full = edge_type_embeddings_full.to(self.device)

        # Now index with edge_types (both are on the same device)
        edge_type_emb = edge_type_embeddings_full[edge_types]

        # Move adjacency mask
        adjacency_mask = self.graph_data.adjacency_mask
        if adjacency_mask is not None and adjacency_mask.device != self.device:
            adjacency_mask = adjacency_mask.to(self.device)

        # Forward pass through GNN
        node_embeddings = self.gnn(
            x=node_embeddings_input,
            edge_index=edge_index,
            edge_type_emb=edge_type_emb,
            edge_weights=edge_weights,
            query_emb=None,  # No specific query in unsupervised training
            adjacency_mask=adjacency_mask
        )

        # Compute losses
        losses = {}

        if self.loss_weights.get('link_prediction', 0) > 0:
            losses['link_prediction'] = self.link_prediction_loss(node_embeddings)

        if self.loss_weights.get('contrastive', 0) > 0:
            losses['contrastive'] = self.contrastive_loss(node_embeddings)

        if self.loss_weights.get('pseudo_query', 0) > 0:
            losses['pseudo_query'] = self.pseudo_query_loss(node_embeddings)

        # Combined loss
        total_loss = sum(
            self.loss_weights.get(k, 0) * v
            for k, v in losses.items()
        )

        # Scale loss for gradient accumulation
        scaled_loss = total_loss / gradient_accumulation_steps

        # Backward pass (accumulate gradients)
        scaled_loss.backward()

        # Only update weights every N steps
        if (self.epoch + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(self.gnn.parameters()) + list(self.link_decoder.parameters()) + list(self.query_matcher.parameters()),
                max_norm=1.0
            )
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Clear GPU cache to free memory
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        # Return loss values
        result = {k: v.item() for k, v in losses.items()}
        result['total'] = total_loss.item()

        return result

    def train(self, num_epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Main training loop.

        Args:
            num_epochs: Number of epochs (uses config if None)

        Returns:
            Training history
        """
        if num_epochs is None:
            num_epochs = self.training_config.get('num_epochs', 100)

        early_stopping_patience = self.training_config.get('early_stopping_patience', 10)
        save_interval = self.training_config.get('save_interval', 5)

        history = {
            'total': [],
            'link_prediction': [],
            'contrastive': [],
            'pseudo_query': []
        }

        logging.info("="*60)
        logging.info(f"Starting training for {num_epochs} epochs")
        logging.info("="*60)

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Train one epoch
            losses = self.train_epoch()

            # Update history
            for k, v in losses.items():
                if k in history:
                    history[k].append(v)

            epoch_time = time.time() - epoch_start

            # Logging
            loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in losses.items()])
            logging.info(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.2f}s) - {loss_str}")

            # Early stopping check
            if losses['total'] < self.best_loss:
                self.best_loss = losses['total']
                self.patience_counter = 0
                self.save_checkpoint('best')
            else:
                self.patience_counter += 1

            if self.patience_counter >= early_stopping_patience:
                logging.info(f"Early stopping triggered at epoch {epoch+1}")
                break

            # Periodic checkpoint
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(f'epoch_{epoch+1}')

            self.epoch = epoch + 1

        logging.info("="*60)
        logging.info("Training complete")
        logging.info(f"Best loss: {self.best_loss:.4f}")
        logging.info("="*60)

        return history

    def save_checkpoint(self, name: str = 'checkpoint'):
        """Save model checkpoint"""
        checkpoint_dir = PROJECT_ROOT / self.reasoning_config.get('training', {}).get('checkpoint_dir', 'data/reasoning/checkpoints')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f'{name}.pt'

        torch.save({
            'epoch': self.epoch,
            'gnn_state_dict': self.gnn.state_dict(),
            'link_decoder_state_dict': self.link_decoder.state_dict(),
            'query_matcher_state_dict': self.query_matcher.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }, checkpoint_path)

        logging.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.gnn.load_state_dict(checkpoint['gnn_state_dict'])
        self.link_decoder.load_state_dict(checkpoint['link_decoder_state_dict'])
        self.query_matcher.load_state_dict(checkpoint['query_matcher_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']

        logging.info(f"Checkpoint loaded from {checkpoint_path} (epoch {self.epoch})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    print("✓ Training module loaded successfully")

