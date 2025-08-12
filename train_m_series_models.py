#!/usr/bin/env python3
"""
Main Training Script for M-Series Euchre AI Models

This script trains the M-Series PyTorch AI models on thousands of games
to achieve optimal weights for the best possible Euchre players.

Usage:
    python train_m_series_models.py [options]

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
Model: Anthropic Claude 3.5 Sonnet
Generation timestamp: 2025-08-12
Context: Creating main training script for M-Series Euchre AI models
"""

import argparse
import os
import sys
import logging
import time
import torch
from pathlib import Path
from typing import Dict, Any

# Add the euchre package to the path
sys.path.insert(0, str(Path(__file__).parent))

from euchre.ai_model.m_series_training import (
    MSeriesSelfPlayTrainer, TrainingConfig
)
from euchre.ai_model.m_series_models import (
    create_mseries_model, create_mseries_risk_profile
)


def setup_logging(log_level: str = "INFO", log_file: str = None) -> None:
    """Setup logging configuration.
    
    Parameters
    ----------
    log_level : str
        Logging level
    log_file : str
        Optional log file path
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger('euchre').setLevel(logging.INFO)
    logging.getLogger('torch').setLevel(logging.WARNING)


def create_training_config(args: argparse.Namespace) -> TrainingConfig:
    """Create training configuration from command line arguments.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
        
    Returns
    -------
    TrainingConfig
        Training configuration object
    """
    return TrainingConfig(
        # Model parameters
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        risk_embedding_size=args.risk_embedding_size,
        
        # Training parameters
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        
        # Self-play parameters
        games_per_epoch=args.games_per_epoch,
        total_training_games=args.total_games,
        
        # Curriculum parameters
        curriculum_stages=args.curriculum_stages,
        games_per_stage=args.games_per_stage,
        
        # Evaluation parameters
        eval_frequency=args.eval_frequency,
        eval_games=args.eval_games,
        
        # Model saving
        save_frequency=args.save_frequency,
        model_dir=args.model_dir,
        
        # Device
        device=args.device
    )


def validate_models(config: TrainingConfig) -> None:
    """Validate that all M-Series models can be created and loaded.
    
    Parameters
    ----------
    config : TrainingConfig
        Training configuration
    """
    logger = logging.getLogger(__name__)
    logger.info("Validating M-Series models...")
    
    try:
        # Test model creation
        model_names = ["magnus", "maverick", "mentor", "mystic"]
        for name in model_names:
            model = create_mseries_model(
                name, 
                config.input_size, 
                config.hidden_size, 
                config.risk_embedding_size
            )
            logger.info(f"✓ {name.capitalize()} model created successfully")
            
            # Test risk profile creation
            risk_profile = create_mseries_risk_profile(name)
            logger.info(f"✓ {name.capitalize()} risk profile created successfully")
            
            # Test forward pass
            test_input = torch.randn(1, config.input_size)
            with torch.no_grad():
                outputs = model(test_input, risk_profile)
            logger.info(f"✓ {name.capitalize()} forward pass successful")
            
    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        raise


def print_training_summary(config: TrainingConfig) -> None:
    """Print a summary of the training configuration.
    
    Parameters
    ----------
    config : TrainingConfig
        Training configuration
    """
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("M-SERIES EUCHRE AI TRAINING CONFIGURATION")
    logger.info("=" * 60)
    
    logger.info("Model Architecture:")
    logger.info(f"  Input Size: {config.input_size}")
    logger.info(f"  Hidden Size: {config.hidden_size}")
    logger.info(f"  Risk Embedding Size: {config.risk_embedding_size}")
    
    logger.info("\nTraining Parameters:")
    logger.info(f"  Epochs: {config.num_epochs}")
    logger.info(f"  Batch Size: {config.batch_size}")
    logger.info(f"  Learning Rate: {config.learning_rate}")
    logger.info(f"  Weight Decay: {config.weight_decay}")
    
    logger.info("\nSelf-Play Configuration:")
    logger.info(f"  Games per Epoch: {config.games_per_epoch}")
    logger.info(f"  Total Training Games: {config.total_training_games}")
    logger.info(f"  Curriculum Stages: {config.curriculum_stages}")
    logger.info(f"  Games per Stage: {config.games_per_stage}")
    
    logger.info("\nEvaluation & Saving:")
    logger.info(f"  Evaluation Frequency: {config.eval_frequency}")
    logger.info(f"  Evaluation Games: {config.eval_games}")
    logger.info(f"  Save Frequency: {config.save_frequency}")
    logger.info(f"  Model Directory: {config.model_dir}")
    
    logger.info("\nDevice:")
    logger.info(f"  Device: {config.device}")
    
    logger.info("=" * 60)


def main():
    """Main training function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train M-Series Euchre AI models on thousands of games"
    )
    
    # Model parameters
    parser.add_argument("--input-size", type=int, default=256,
                       help="Input feature vector size (default: 256)")
    parser.add_argument("--hidden-size", type=int, default=512,
                       help="Hidden layer size (default: 512)")
    parser.add_argument("--risk-embedding-size", type=int, default=64,
                       help="Risk parameter embedding size (default: 64)")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=1000,
                       help="Number of training epochs (default: 1000)")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Training batch size (default: 32)")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="Learning rate (default: 0.001)")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                       help="Weight decay (default: 1e-5)")
    
    # Self-play parameters
    parser.add_argument("--games-per-epoch", type=int, default=100,
                       help="Games per epoch (default: 100)")
    parser.add_argument("--total-games", type=int, default=10000,
                       help="Total training games (default: 10000)")
    
    # Curriculum parameters
    parser.add_argument("--curriculum-stages", type=int, default=5,
                       help="Number of curriculum stages (default: 5)")
    parser.add_argument("--games-per-stage", type=int, default=2000,
                       help="Games per curriculum stage (default: 2000)")
    
    # Evaluation parameters
    parser.add_argument("--eval-frequency", type=int, default=100,
                       help="Evaluation frequency in epochs (default: 100)")
    parser.add_argument("--eval-games", type=int, default=50,
                       help="Number of games for evaluation (default: 50)")
    
    # Model saving
    parser.add_argument("--save-frequency", type=int, default=500,
                       help="Model save frequency in epochs (default: 500)")
    parser.add_argument("--model-dir", type=str, default="trained_models",
                       help="Directory to save trained models (default: trained_models)")
    
    # Device
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use: auto, cpu, or cuda (default: auto)")
    
    # Logging
    parser.add_argument("--log-level", type=str, default="INFO",
                       help="Logging level (default: INFO)")
    parser.add_argument("--log-file", type=str, default=None,
                       help="Log file path (optional)")
    
    # Validation only
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate models, don't train")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting M-Series Euchre AI Training")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    
    try:
        # Create training configuration
        config = create_training_config(args)
        
        # Print training summary
        print_training_summary(config)
        
        # Validate models
        validate_models(config)
        
        if args.validate_only:
            logger.info("Model validation completed successfully. Exiting.")
            return
        
        # Create trainer
        logger.info("Creating M-Series trainer...")
        trainer = MSeriesSelfPlayTrainer(config)
        
        # Generate training data
        logger.info("Generating training data through self-play...")
        start_time = time.time()
        training_data = trainer.generate_training_data()
        data_generation_time = time.time() - start_time
        
        logger.info(f"Training data generation completed in {data_generation_time:.2f} seconds")
        logger.info(f"Generated {len(training_data)} training games")
        
        # Train models
        logger.info("Starting model training...")
        start_time = time.time()
        trainer.train_models(training_data)
        training_time = time.time() - start_time
        
        logger.info(f"Model training completed in {training_time:.2f} seconds")
        
        # Save training history
        logger.info("Saving training history...")
        trainer.save_training_history()
        
        # Final summary
        total_time = data_generation_time + training_time
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"Data generation: {data_generation_time:.2f} seconds")
        logger.info(f"Model training: {training_time:.2f} seconds")
        logger.info(f"Models saved to: {config.model_dir}")
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main() 