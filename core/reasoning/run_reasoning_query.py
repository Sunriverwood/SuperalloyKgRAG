"""
Inference Script for Graph Reasoning Model

Usage:
    python core/reasoning/run_reasoning_query.py --query "What is the relationship between nickel and turbine blades?"
"""

import argparse
import logging
from pathlib import Path
import yaml
import json

from core.query_qwen.reasoning_query_qwen import ReasoningQueryHandler


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def setup_logging():
    """Setup logging configuration"""
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "reasoning_query.log", mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def load_config():
    """Load configuration"""
    config_path = PROJECT_ROOT / "config" / "settings.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run Graph Reasoning Query")
    parser.add_argument('--query', type=str, required=True, help='Natural language query')
    parser.add_argument('--method', type=str, default='ppr', choices=['ppr', 'gnn'],
                       help='Reasoning method (ppr or gnn)')
    parser.add_argument('--no-llm', action='store_true', help='Skip LLM answer generation')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file path')
    args = parser.parse_args()

    # Setup
    setup_logging()
    config = load_config()

    logging.info("="*80)
    logging.info("Graph Reasoning Query System")
    logging.info("="*80)

    # Load handler (with trained model)
    logging.info("Loading reasoning model...")
    handler = ReasoningQueryHandler(config, load_trained_model=True)

    # Run query
    logging.info(f"Processing query: {args.query}")
    results = handler.query(
        query_text=args.query,
        method=args.method,
        include_llm_answer=not args.no_llm
    )

    # Print results
    print("\n" + "=" * 80)
    logging.info("=" * 80)
    logging.info("REASONING RESULTS")
    print("REASONING RESULTS")
    logging.info("=" * 80)
    print(f"\nQuery: {results['query']}")
    logging.info(f"Query: {results['query']}")

    print(f"\n{'Top Relevant Entities:':<30}")
    logging.info("Top Relevant Entities:")
    print("-" * 80)
    logging.info("-" * 80)
    for i, node in enumerate(results['top_nodes'][:10], 1):
        line = f"{i:2}. {node['name']:<50} (score: {node['score']:.4f})"
        print(line)
        logging.info(line)

    print(f"\n{'Reasoning Paths:':<30}")
    logging.info("Reasoning Paths:")
    print("-" * 80)
    logging.info("-" * 80)
    if results['paths']:
        for i, path_info in enumerate(results['paths'][:5], 1):
            header = f"\nPath {i} (confidence: {path_info['score']:.4f}):"
            print(header)
            logging.info(header)
            print(path_info['explanation'])
            logging.info(path_info.get('explanation', ''))
    else:
        print("No reasoning paths found.")
        logging.info("No reasoning paths found.")

    if 'answer' in results:
        print(f"\n{'Final Answer:':<30}")
        logging.info("Final Answer:")
        print("-" * 80)
        logging.info("-" * 80)
        print(results['answer'])
        logging.info(results['answer'])

    print("\n" + "=" * 80)
    logging.info("=" * 80)

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logging.info(f"Results saved to {output_path}")

    logging.info("Query processing complete")

if __name__ == "__main__":
    main()

