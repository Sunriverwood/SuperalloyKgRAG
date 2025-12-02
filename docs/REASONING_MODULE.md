# Graph Reasoning Module

## Overview

This module implements an **unsupervised graph reasoning system** for knowledge graph-based question answering. It uses query-aware graph neural networks (RGAT) with self-supervised training to enable multi-hop reasoning and path extraction.

## Architecture

### Key Components

1. **Data Loader** (`core/reasoning/data_loader.py`)
   - Loads knowledge graph from `final_graph.json`
   - Extracts embeddings from `embedding.db` (LanceDB)
   - Builds edge indices with composite_importance weights
   - Creates adjacency masks for graph constraints

2. **Query-Aware RGAT** (`core/reasoning/models/rgat.py`)
   - Relational Graph Attention Network with query awareness
   - Multi-head attention with edge type and weight integration
   - Respects graph structure constraints via adjacency masking

3. **Self-Supervised Trainer** (`core/reasoning/training/trainer.py`)
   - **Link Prediction**: Edge reconstruction with weighted loss
   - **Contrastive Learning**: InfoNCE for node representation
   - **Pseudo Query Matching**: Query-entity similarity learning

4. **Reasoning Engine** (`core/reasoning/inference/reasoner.py`)
   - Node scoring with query-aware GNN
   - Personalized PageRank (PPR) propagation
   - Path extraction and ranking on real graph structure

5. **Query Handler** (`core/query_qwen/reasoning_query_qwen.py`)
   - Unified interface for reasoning queries
   - LLM answer generation with reasoning context
   - Integration with existing query system

## Installation

### Dependencies

```bash
pip install torch torchvision
pip install networkx numpy pandas
pip install lancedb pyarrow
pip install openai  # For API compatibility
pip install pyyaml
```

## Configuration

The reasoning system is configured in `config/settings.yaml` under the `reasoning` section:

```yaml
reasoning:
  model:
    hidden_dim: 256
    num_layers: 3
    num_heads: 4
    dropout: 0.1
    use_edge_weights: true  # Use composite_importance
  
  training:
    num_epochs: 100
    learning_rate: 0.001
    batch_size: 512
    loss_weights:
      link_prediction: 1.0
      contrastive: 0.5
      pseudo_query: 1.0
  
  inference:
    top_k_nodes: 20
    ppr_alpha: 0.15
    max_path_length: 3
    max_paths_per_query: 10
```

## Usage

### 1. Training the Model

Train the graph reasoning model using self-supervised objectives:

```bash
# Basic training
python core/reasoning/train_reasoning.py --epochs 100 --device cuda

# With custom parameters
python core/reasoning/train_reasoning.py --epochs 50 --batch-size 1024 --lr 0.0005
```

The trained model will be saved to `data/reasoning/model.pt`.

### 2. Running Inference

Query the knowledge graph with natural language:

```bash
# Basic query
python core/reasoning/run_reasoning_query.py --query "What is the relationship between nickel-based superalloys and turbine blades?"

# Using different reasoning method
python core/reasoning/run_reasoning_query.py --query "How does temperature affect creep resistance?" --method gnn

# Save results to file
python core/reasoning/run_reasoning_query.py --query "What are the properties of Inconel 718?" --output results.json
```

### 3. Programmatic Usage

```python
from core.query_qwen.reasoning_query_qwen import ReasoningQueryHandler, load_config

# Load configuration
config = load_config()

# Create handler (loads trained model)
handler = ReasoningQueryHandler(config, load_trained_model=True)

# Run query
results = handler.query(
    query_text="What materials are used in turbine blades?",
    method='ppr',
    include_llm_answer=True
)

# Access results
print(f"Top entities: {results['top_nodes']}")
print(f"Reasoning paths: {results['paths']}")
print(f"Answer: {results['answer']}")
```

## How It Works

### Self-Supervised Training

The model learns from the graph structure without labeled data:

1. **Link Prediction**: Predicts whether edges exist between node pairs
   - Positive samples: Existing edges (weighted by composite_importance)
   - Negative samples: Random non-existing edges
   
2. **Graph Contrastive Learning**: Creates two augmented views and maximizes agreement
   - Helps learn robust node representations
   
3. **Pseudo Query Task**: For each triplet (head, relation, tail):
   - Query = encode(head + relation)
   - Positive = tail entity
   - Negatives = random entities
   - Trains query-entity matching function

### Inference Process

1. **Query Encoding**: Convert natural language to embedding vector

2. **Node Scoring**: 
   - Pass through query-aware GNN
   - Compute similarity with query using trained matcher
   - Get top-k relevant entities

3. **Graph Propagation** (PPR method):
   - Initialize distribution based on query scores
   - Propagate: π_{k+1} = α·π_0 + (1-α)·π_k·P
   - Transition matrix P uses composite_importance weights
   - Respects graph constraints (only real edges)

4. **Path Extraction**:
   - BFS from top start nodes to top end nodes
   - Only traverse real edges in graph
   - Score paths by composite_importance × attention

5. **Answer Generation**:
   - Format paths and entities as context
   - Send to LLM for natural language answer

## Graph Constraints

The system enforces **strict graph constraints**:

- **Attention Masking**: Non-existing edges get -�?attention
- **PPR Transition**: P[i,j] = 0 if edge (i,j) doesn't exist
- **Path Search**: BFS only along real edges in NetworkX graph
- **Path Score**: I(π) = 0 if any edge in path is not real

This ensures all reasoning is **grounded in the actual knowledge graph structure**.

## Edge Weights

The system leverages **composite_importance** scores:

- Extracted from `final_graph.json` edge attributes
- Used as attention priors in RGAT layers
- Weights positive samples in link prediction loss
- Defines transition probabilities in PPR
- Combined with learned attention for path scoring

## Output Format

Query results include:

```python
{
    "query": "original query text",
    "top_nodes": [
        {
            "id": "node_id",
            "name": "entity name",
            "score": 0.95
        },
        ...
    ],
    "paths": [
        {
            "path": ["node1", "node2", "node3"],
            "score": 0.87,
            "edge_types": ["relation1", "relation2"],
            "explanation": "formatted path with scores"
        },
        ...
    ],
    "num_paths": 10,
    "answer": "LLM-generated answer based on paths"
}
```

## Utilities

The `utils/graph_reasoning_utils.py` provides:

- `apply_adjacency_mask()`: Apply graph constraints to scores
- `normalize_edge_weights()`: Normalize composite_importance values
- `score_path_by_importance()`: Score paths with importance + attention
- `extract_paths_bfs()`: BFS path extraction on real graph
- `rank_paths()`: Rank paths by combined scores
- `format_path_explanation()`: Human-readable path formatting
- `PseudoQueryGenerator`: Generate training pseudo queries

## Module Structure

```
core/reasoning/
├── __init__.py
├── data_loader.py          # Graph & embedding loading
├── models/
�?  ├── __init__.py
�?  └── rgat.py            # Query-aware RGAT model
├── training/
�?  ├── __init__.py
�?  └── trainer.py         # Self-supervised training
└── inference/
    ├── __init__.py
    └── reasoner.py        # Reasoning engine (PPR, paths)

core/query_qwen/
└── reasoning_query_qwen.py # Query handler integration

utils/
└── graph_reasoning_utils.py # Helper functions


├── train_reasoning.py      # Training script
└── run_reasoning_query.py  # Inference script
```

## Performance Tips

1. **GPU Acceleration**: Use `--device cuda` for training (10-100x speedup)

2. **Batch Size**: Adjust based on available memory
   - Larger batches (1024-2048) for better convergence
   - Smaller batches (256-512) if memory limited

3. **Edge Sampling**: For very large graphs, consider edge sampling during training

4. **Path Limits**: Adjust `max_path_length` and `max_paths_per_query` based on query complexity

5. **PPR Parameters**: 
   - Lower `alpha` (0.1) for more global exploration
   - Higher `alpha` (0.3) for more local focus

## Troubleshooting

**Issue**: Model not converging
- Check learning rate (try 0.0001-0.001 range)
- Verify edge weights are normalized properly
- Ensure sufficient training epochs (50-100 minimum)

**Issue**: No paths found
- Check if start and end nodes are connected in graph
- Increase `max_path_length` parameter
- Verify adjacency mask is correct

**Issue**: Out of memory
- Reduce batch size
- Use smaller hidden dimension
- Reduce number of layers or heads

**Issue**: Query encoding fails
- Verify QWEN_API_KEY environment variable is set
- Check network connectivity
- Ensure embedding model matches config

## Future Enhancements

- [ ] Attention visualization
- [ ] Multi-GPU training support
- [ ] Path diversity metrics
- [ ] Interactive path exploration UI
- [ ] Query expansion techniques
- [ ] Temporal reasoning support

## References

- **RGAT**: Relational Graph Attention Networks
- **PPR**: Personalized PageRank for graph reasoning
- **InfoNCE**: Contrastive learning objective
- **BPR**: Bayesian Personalized Ranking for recommendations

## License

Part of the SuperalloyKgRAG project.

