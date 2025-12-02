"""
Reasoning Query Handler

Integrates graph reasoning capabilities with the existing query system.
Provides a unified interface for query-aware graph reasoning.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
import torch
import numpy as np
from openai import OpenAI

from core.reasoning.data_loader import GraphReasoningDataLoader, GraphData
from core.reasoning.models.rgat import QueryAwareRGAT
from core.reasoning.training.trainer import GraphReasoningTrainer, QueryEntityMatcher
from core.reasoning.inference.reasoner import GraphReasoner


# --- 项目根目录定义 ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def setup_logging(config: Dict[str, Any]):
    """根据配置文件设置日志记录器"""
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
    relative_log_path = log_config.get("log_file", "logs/superalloyKgRAG.log")
    log_file = PROJECT_ROOT / relative_log_path

    log_file.parent.mkdir(exist_ok=True, parents=True)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info("推理查询日志记录器设置完成")


def load_config(settings_filename: str = "settings.yaml") -> Dict[str, Any]:
    """加载YAML配置文件"""
    config_path = PROJECT_ROOT / "config" / settings_filename
    logging.info(f"正在从 {config_path} 加载配置...")
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件 {config_path} 未找到！")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    logging.info("配置加载成功。")
    return config


class ReasoningQueryHandler:
    """
    Query handler with graph reasoning capabilities.

    Workflow:
    1. Query encoding (using text encoder)
    2. Node retrieval (initial candidates)
    3. Graph reasoning (PPR propagation + path extraction)
    4. Answer generation (LLM synthesis with reasoning paths)
    """

    def __init__(self, config: Dict[str, Any], load_trained_model: bool = True):
        """
        Args:
            config: Configuration dictionary
            load_trained_model: Whether to load pre-trained model (False for training mode)
        """
        self.config = config
        self.reasoning_config = config.get("reasoning", {})

        # Setup API client for text encoding and generation
        self.api_key = os.getenv("QWEN_API_KEY")
        if not self.api_key:
            logging.warning("未找到 QWEN_API_KEY 环境变量")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        self.embedding_model_name = config["embedding"]["model"]
        self.generation_model_name = config["query"]["generation_model"]
        self.temperature = config["query"]["temperature"]

        # Device setup
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"Using device: {self.device}")

        # Load graph data
        logging.info("Loading graph data for reasoning...")
        self.data_loader = GraphReasoningDataLoader(config)
        self.graph_data = self.data_loader.load(device=self.device)

        # Initialize or load models
        if load_trained_model:
            self._load_trained_models()
        else:
            self._initialize_models()

        # Create reasoner
        self.reasoner = GraphReasoner(
            config=config,
            graph_data=self.graph_data,
            gnn_model=self.gnn,
            query_matcher=self.query_matcher,
            device=self.device
        )

        logging.info("ReasoningQueryHandler initialized successfully")

    def _initialize_models(self):
        """Initialize models for training"""
        model_config = self.reasoning_config.get('model', {})

        self.gnn = QueryAwareRGAT(
            input_dim=self.graph_data.embed_dim,
            hidden_dim=model_config.get('hidden_dim', 256),
            output_dim=model_config.get('hidden_dim', 256),
            num_layers=model_config.get('num_layers', 3),
            num_heads=model_config.get('num_heads', 4),
            dropout=model_config.get('dropout', 0.1),
            use_edge_weights=model_config.get('use_edge_weights', True),
            query_dim=self.graph_data.embed_dim,
            edge_type_dim=self.graph_data.embed_dim  # Edge type embeddings stay in original dimension
        ).to(self.device)

        self.query_matcher = QueryEntityMatcher(
            query_dim=self.graph_data.embed_dim,
            entity_dim=model_config.get('hidden_dim', 256),
            hidden_dim=model_config.get('hidden_dim', 256)
        ).to(self.device)

        logging.info("Models initialized for training")

    def _load_trained_models(self):
        """Load pre-trained models"""
        model_path = PROJECT_ROOT / self.reasoning_config.get('output', {}).get('model_path', 'data/reasoning/model.pt')

        if not model_path.exists():
            logging.warning(f"Trained model not found at {model_path}, initializing new models")
            self._initialize_models()
            return

        logging.info(f"Loading trained model from {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)

        # Initialize models with same architecture
        self._initialize_models()

        # Load state dicts
        self.gnn.load_state_dict(checkpoint['gnn_state_dict'])
        self.query_matcher.load_state_dict(checkpoint['query_matcher_state_dict'])

        # Set to eval mode
        self.gnn.eval()
        self.query_matcher.eval()

        logging.info(f"Loaded trained model from epoch {checkpoint.get('epoch', 'unknown')}")

    def encode_query(self, query_text: str) -> np.ndarray:
        """
        Encode query text using embedding model.

        Args:
            query_text: Natural language query

        Returns:
            Query embedding vector
        """
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model_name,
                input=query_text,
                dimensions=self.graph_data.embed_dim
            )

            embedding = np.array(response.data[0].embedding)
            return embedding

        except Exception as e:
            logging.error(f"Query encoding failed: {e}")
            # Fallback to zero vector
            return np.zeros(self.graph_data.embed_dim)

    def generate_answer(self, query: str, reasoning_results: Dict[str, Any]) -> str:
        """
        Generate final answer using LLM with reasoning context.

        Args:
            query: Original query
            reasoning_results: Results from graph reasoning

        Returns:
            Generated answer
        """
        # Format reasoning paths as context
        context_parts = []

        # Add top nodes
        context_parts.append("## Relevant Entities:")
        for node_info in reasoning_results['top_nodes'][:5]:
            context_parts.append(f"- {node_info['name']} (relevance: {node_info['score']:.3f})")

        # Add reasoning paths
        if reasoning_results['paths']:
            context_parts.append("\n## Reasoning Paths:")
            for i, path_info in enumerate(reasoning_results['paths'][:3], 1):
                context_parts.append(f"\nPath {i} (confidence: {path_info['score']:.3f}):")
                context_parts.append(path_info['explanation'])

        context = "\n".join(context_parts)

        # Create prompt
        prompt = f"""Based on the following knowledge graph reasoning results, please answer the question.

Question: {query}

{context}

Please provide a comprehensive answer based on the reasoning paths and entities above. 
Explain how the paths support your answer."""

        try:
            response = self.client.chat.completions.create(
                model=self.generation_model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on knowledge graph reasoning."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature
            )

            answer = response.choices[0].message.content
            return answer

        except Exception as e:
            logging.error(f"Answer generation failed: {e}")
            return "Error generating answer. Please try again."

    def query(self, query_text: str, method: str = 'ppr',
             include_llm_answer: bool = True) -> Dict[str, Any]:
        """
        Main query interface.

        Args:
            query_text: Natural language query
            method: Reasoning method ('ppr' or 'gnn')
            include_llm_answer: Whether to generate final answer with LLM

        Returns:
            Complete query results including reasoning and answer
        """
        logging.info("="*80)
        logging.info(f"Processing query: {query_text}")
        logging.info("="*80)

        # 1. Encode query
        query_embedding_np = self.encode_query(query_text)
        query_embedding = torch.from_numpy(query_embedding_np).float().to(self.device)

        # 2. Perform graph reasoning
        reasoning_results = self.reasoner.reason(
            query_text=query_text,
            text_encoder=self.encode_query,
            method=method
        )

        # 3. Generate final answer (optional)
        if include_llm_answer:
            answer = self.generate_answer(query_text, reasoning_results)
            reasoning_results['answer'] = answer

        logging.info(f"Query processing complete")
        logging.info(f"  Top nodes: {len(reasoning_results['top_nodes'])}")
        logging.info(f"  Reasoning paths: {reasoning_results['num_paths']}")
        logging.info("="*80)

        return reasoning_results

    def train(self, num_epochs: Optional[int] = None):
        """
        Train the reasoning model.

        Args:
            num_epochs: Number of training epochs (uses config if None)
        """
        logging.info("Starting graph reasoning model training...")

        trainer = GraphReasoningTrainer(
            config=self.config,
            graph_data=self.graph_data,
            device=self.device
        )

        # Train
        history = trainer.train(num_epochs=num_epochs)

        # Save final model
        model_path = PROJECT_ROOT / self.reasoning_config.get('output', {}).get('model_path', 'data/reasoning/model.pt')
        model_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'epoch': trainer.epoch,
            'gnn_state_dict': trainer.gnn.state_dict(),
            'link_decoder_state_dict': trainer.link_decoder.state_dict(),
            'query_matcher_state_dict': trainer.query_matcher.state_dict(),
            'best_loss': trainer.best_loss,
            'config': self.config,
            'history': history
        }, model_path)

        logging.info(f"Training complete. Model saved to {model_path}")

        # Update local references
        self.gnn = trainer.gnn
        self.query_matcher = trainer.query_matcher

        # Recreate reasoner with trained models
        self.reasoner = GraphReasoner(
            config=self.config,
            graph_data=self.graph_data,
            gnn_model=self.gnn,
            query_matcher=self.query_matcher,
            device=self.device
        )

        return history


def print_results(results: Dict[str, Any]):
    """
    Format and print reasoning results.

    Args:
        results: Reasoning results dictionary
    """
    print("\n" + "="*80)
    print("REASONING RESULTS")
    print("="*80)
    print(f"\nQuery: {results['query']}")

    # Top entities
    print(f"\n{'Top Relevant Entities:':<30}")
    print("-" * 80)
    for i, node in enumerate(results['top_nodes'][:10], 1):
        print(f"{i:2}. {node['name']:<50} (score: {node['score']:.4f})")

    # Reasoning paths
    print(f"\n{'Reasoning Paths:':<30}")
    print("-" * 80)
    if results['paths']:
        for i, path_info in enumerate(results['paths'][:5], 1):
            print(f"\nPath {i} (confidence: {path_info['score']:.4f}):")
            print(path_info['explanation'])
    else:
        print("No reasoning paths found.")

    # Final answer
    if 'answer' in results:
        print(f"\n{'Final Answer:':<30}")
        print("-" * 80)
        print(results['answer'])

    print("\n" + "="*80)


def interactive_mode(config: Dict[str, Any]):
    """
    Interactive mode for reasoning queries.

    Args:
        config: Configuration dictionary
    """
    print("\n" + "="*80)
    print("Graph Reasoning Query System - Interactive Mode")
    print("="*80)

    # Check if model exists
    model_path = PROJECT_ROOT / config.get('reasoning', {}).get('output', {}).get('model_path', 'data/reasoning/model.pt')
    model_exists = model_path.exists()

    if model_exists:
        print(f"\n✓ Found trained model: {model_path}")
        load_model = True
    else:
        print(f"\n⚠ No trained model found at: {model_path}")
        train_choice = input("Do you want to train the model now? (yes/no) [yes]: ").strip().lower()

        if train_choice in ['', 'yes', 'y']:
            epochs_input = input("Number of training epochs [50]: ").strip()
            num_epochs = int(epochs_input) if epochs_input else 50

            print("\n" + "="*80)
            print("Starting Model Training...")
            print("="*80)

            # Train model
            handler = ReasoningQueryHandler(config, load_trained_model=False)
            handler.train(num_epochs=num_epochs)
            load_model = True

            print("\n✓ Training complete!")
        else:
            print("\n⚠ Cannot proceed without a trained model. Exiting.")
            return

    # Load handler
    print("\nLoading reasoning model...")
    handler = ReasoningQueryHandler(config, load_trained_model=load_model)
    print("✓ Model loaded successfully!\n")

    # Query loop
    while True:
        print("\n" + "-" * 80)
        query = input("\nEnter your query (or 'quit' to exit): ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break

        if not query:
            print("⚠ Please enter a valid query.")
            continue

        # Choose method
        method_input = input("Reasoning method (ppr/gnn) [ppr]: ").strip().lower()
        method = method_input if method_input in ['ppr', 'gnn'] else 'ppr'

        # Choose whether to generate answer
        llm_input = input("Generate LLM answer? (yes/no) [yes]: ").strip().lower()
        include_llm = llm_input not in ['no', 'n']

        # Save to file?
        save_input = input("Save results to file? (yes/no) [no]: ").strip().lower()
        output_file = None
        if save_input in ['yes', 'y']:
            output_file = input("Output file path [data/reasoning/query_result.json]: ").strip()
            if not output_file:
                output_file = "data/reasoning/query_result.json"

        # Run query
        print("\n" + "=" * 80)
        print("Processing query...")
        print("=" * 80)

        try:
            results = handler.query(
                query_text=query,
                method=method,
                include_llm_answer=include_llm
            )

            # Print results
            print_results(results)

            # Write human-readable results to log
            log_lines = []
            log_lines.append("\n" + "=" * 80)
            log_lines.append("REASONING RESULTS")
            log_lines.append("=" * 80)
            log_lines.append(f"Query: {results.get('query', query)}")
            log_lines.append("\nTop Relevant Entities:")
            for i, node in enumerate(results.get('top_nodes', [])[:10], 1):
                log_lines.append(f"{i}. {node.get('name', '<unknown>')} (score: {node.get('score', 0):.4f})")

            log_lines.append("\nReasoning Paths:")
            paths = results.get('paths', [])
            if paths:
                for i, path_info in enumerate(paths[:5], 1):
                    log_lines.append(f"\nPath {i} (confidence: {path_info.get('score', 0):.4f}):")
                    log_lines.append(path_info.get('explanation', '').strip())
            else:
                log_lines.append("No reasoning paths found.")

            if 'answer' in results:
                log_lines.append("\nFinal Answer:")
                log_lines.append(results.get('answer', '').strip())

            logging.info("\n".join(log_lines))

            # Also log full JSON for traceability
            logging.info("Reasoning results (JSON):\n%s", json.dumps(results, ensure_ascii=False, indent=2))

            # Save if requested
            if output_file:
                output_path = PROJECT_ROOT / output_file
                output_path.parent.mkdir(parents=True, exist_ok=True)

                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

                print(f"\n✓ Results saved to: {output_path}")

        except Exception as e:
            print(f"\n✗ Error processing query: {e}")
            logging.error(f"Query processing error: {e}", exc_info=True)


def command_line_mode():
    """
    Command-line mode with argument parsing.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Graph Reasoning Query System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Query mode
  python reasoning_query_qwen.py --query "What is nickel used for?"
  
  # Interactive mode
  python reasoning_query_qwen.py --interactive
  
  # Training mode
  python reasoning_query_qwen.py --train --epochs 100
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--interactive', '-i', action='store_true',
                           help='Run in interactive mode')
    mode_group.add_argument('--train', action='store_true',
                           help='Train the model (skip if model exists)')

    # Query parameters
    parser.add_argument('--query', '-q', type=str, default=None,
                       help='Natural language query')
    parser.add_argument('--method', '-m', type=str, default='ppr',
                       choices=['ppr', 'gnn'],
                       help='Reasoning method (default: ppr)')
    parser.add_argument('--no-llm', action='store_true',
                       help='Skip LLM answer generation')

    # Training parameters
    parser.add_argument('--epochs', '-e', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--force-train', action='store_true',
                       help='Force training even if model exists')

    # Output
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Save results to JSON file')

    args = parser.parse_args()

    # Load config
    config = load_config()
    setup_logging(config)

    # Check model existence
    model_path = PROJECT_ROOT / config.get('reasoning', {}).get('output', {}).get('model_path', 'data/reasoning/model.pt')
    model_exists = model_path.exists()

    # Interactive mode
    if args.interactive:
        interactive_mode(config)
        return

    # Training mode
    if args.train or args.force_train:
        if model_exists and not args.force_train:
            print(f"\n✓ Model already exists at: {model_path}")
            print("  Use --force-train to retrain anyway.")
            print("  Skipping training.\n")
        else:
            print("\n" + "="*80)
            print("Starting Model Training...")
            print("="*80)

            handler = ReasoningQueryHandler(config, load_trained_model=False)
            handler.train(num_epochs=args.epochs)

            print("\n✓ Training complete!")
        return

    # Query mode
    if args.query:
        print("\n" + "="*80)
        print("Graph Reasoning Query System")
        print("="*80)

        if not model_exists:
            print(f"\n✗ Error: No trained model found at {model_path}")
            print("  Please train the model first using: --train")
            return

        # Load handler
        print("\nLoading reasoning model...")
        handler = ReasoningQueryHandler(config, load_trained_model=True)

        # Run query
        print(f"\nProcessing query: {args.query}")
        results = handler.query(
            query_text=args.query,
            method=args.method,
            include_llm_answer=not args.no_llm
        )

        # Print results
        print_results(results)

        # Save if requested
        if args.output:
            output_path = PROJECT_ROOT / args.output
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            print(f"\n✓ Results saved to: {output_path}")
    else:
        # No arguments - show help or enter interactive
        print("\nNo query provided. Starting interactive mode...\n")
        interactive_mode(config)


def main():
    """Main entry point"""
    try:
        command_line_mode()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        logging.error(f"Fatal error: {e}", exc_info=True)


if __name__ == "__main__":
    main()

