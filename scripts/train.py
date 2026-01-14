#!/usr/bin/env python3
"""
Training Script for CommunityMed AI
Fine-tune MedGemma on TB X-ray datasets using QLoRA
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger


def setup_logging(log_dir: str = "./logs"):
    """Setup logging configuration"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger.add(
        f"{log_dir}/training_{{time}}.log",
        rotation="100 MB",
        level="INFO",
    )


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Fine-tune MedGemma for TB detection"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="google/medgemma-4b-it",
        help="Base model to fine-tune",
    )
    
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["shenzhen", "montgomery"],
        help="Datasets to use for training",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints/tb_finetuned",
        help="Output directory for model",
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size",
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank",
    )
    
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha",
    )
    
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        default=True,
        help="Use 4-bit quantization",
    )
    
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Use Weights & Biases logging",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run without actually training",
    )
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    setup_logging()
    logger.info("=" * 60)
    logger.info("CommunityMed AI - Training Script")
    logger.info("=" * 60)
    logger.info(f"Config: {vars(args)}")
    
    # Check for GPU
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            device = "cpu"
            logger.warning("No GPU available, training on CPU (will be slow)")
    except ImportError:
        logger.error("PyTorch not installed")
        return 1
    
    if args.dry_run:
        logger.info("Dry run mode - not actually training")
        logger.info("Would train with:")
        logger.info(f"  Model: {args.model_name}")
        logger.info(f"  Datasets: {args.datasets}")
        logger.info(f"  Epochs: {args.epochs}")
        logger.info(f"  Batch size: {args.batch_size}")
        logger.info(f"  Output: {args.output_dir}")
        return 0
    
    # Import training components
    try:
        from data import TBDatasetLoader
        from models import MedGemmaLoader
        from models.fine_tuning import FineTuner, TrainingConfig
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure all dependencies are installed: pip install -r requirements.txt")
        return 1
    
    # Step 1: Load datasets
    logger.info("Step 1: Loading datasets...")
    loader = TBDatasetLoader("./data")
    
    train_data, val_data = loader.create_training_data(
        datasets=args.datasets,
        train_ratio=0.8,
    )
    logger.info(f"Training samples: {len(train_data)}")
    logger.info(f"Validation samples: {len(val_data)}")
    
    # Step 2: Load model
    logger.info("Step 2: Loading MedGemma model...")
    model_loader = MedGemmaLoader(
        model_name=args.model_name,
        use_4bit=args.use_4bit,
        device=device,
    )
    model, processor = model_loader.load_multimodal_model()
    
    # Step 3: Setup fine-tuner
    logger.info("Step 3: Setting up fine-tuner...")
    config = TrainingConfig(
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_wandb=args.wandb,
    )
    
    fine_tuner = FineTuner(
        model=model,
        processor=processor,
        config=config,
    )
    
    # Step 4: Train
    logger.info("Step 4: Starting training...")
    fine_tuner.train(train_data, val_data)
    
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Model saved to: {args.output_dir}")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
