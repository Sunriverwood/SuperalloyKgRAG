"""
Training Script for Graph Reasoning Model

Usage:
    python core/reasoning/train_reasoning.py [--epochs 100] [--device cuda]
"""

import argparse
import logging
from pathlib import Path
import yaml

from core.reasoning.data_loader import GraphReasoningDataLoader
from core.reasoning.training.trainer import GraphReasoningTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def setup_logging():
    """Setup logging configuration"""
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "train_reasoning.log", mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def load_config():
    """Load configuration"""
    config_path = PROJECT_ROOT / "config" / "settings.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train Graph Reasoning Model")
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device to use')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    args = parser.parse_args()

    # Setup
    setup_logging()
    config = load_config()

    logging.info("="*80)
    logging.info("Graph Reasoning Model Training")
    logging.info("="*80)

    # Override config with command line args
    if args.batch_size is not None:
        config['reasoning']['training']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['reasoning']['training']['learning_rate'] = args.lr

    # Load data
    logging.info("Loading graph data...")
    data_loader = GraphReasoningDataLoader(config)
    graph_data = data_loader.load(device=args.device)

    # Create trainer
    logging.info("Initializing trainer...")
    trainer = GraphReasoningTrainer(config, graph_data, device=args.device)

    # Train
    logging.info("Starting training...")
    history = trainer.train(num_epochs=args.epochs)

    # Save final model
    output_path = PROJECT_ROOT / config['reasoning']['output']['model_path']
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import torch
    torch.save({
        'epoch': trainer.epoch,
        'gnn_state_dict': trainer.gnn.state_dict(),
        'link_decoder_state_dict': trainer.link_decoder.state_dict(),
        'query_matcher_state_dict': trainer.query_matcher.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'best_loss': trainer.best_loss,
        'config': config,
        'history': history
    }, output_path)

    logging.info("="*80)
    logging.info(f"Training complete! Model saved to {output_path}")
    logging.info(f"Best loss: {trainer.best_loss:.4f}")
    logging.info("="*80)


if __name__ == "__main__":
    main()

