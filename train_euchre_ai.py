#!/usr/bin/env python3
"""Main training script for the euchre AI model.

This script trains a neural network to play euchre using game data.
"""

import argparse
import os
import logging
from pathlib import Path

from euchre.ai_model import EuchreNN, TrainingPipeline, ModelEvaluator


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration.
    
    Parameters
    ----------
    log_level : str
        Logging level
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train euchre AI model")
    parser.add_argument("--data-dir", required=True, help="Directory containing training data")
    parser.add_argument("--output-dir", default="models", help="Directory to save trained models")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden-size", type=int, default=256, help="Hidden layer size")
    parser.add_argument("--generate-data", action="store_true", help="Generate sample training data")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model after training")
    parser.add_argument("--num-eval-games", type=int, default=100, help="Number of games for evaluation")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create model
    logger.info("Creating neural network model...")
    model = EuchreNN(
        input_size=128,
        hidden_size=args.hidden_size,
        output_size=64,
        risk_embedding_size=32,
        use_risk_attention=True
    )
    
    # Create training pipeline
    logger.info("Setting up training pipeline...")
    pipeline = TrainingPipeline(
        model=model,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Generate sample data if requested
    if args.generate_data:
        logger.info("Generating sample training data...")
        pipeline.generate_sample_data(num_games=1000)
    
    # Train the model
    logger.info("Starting training...")
    model_save_path = os.path.join(args.output_dir, "euchre_model.pth")
    
    history = pipeline.train(
        num_epochs=args.epochs,
        validation_split=0.2,
        save_path=model_save_path
    )
    
    logger.info(f"Training complete! Model saved to {model_save_path}")
    
    # Evaluate the model if requested
    if args.evaluate:
        logger.info("Evaluating trained model...")
        
        evaluator = ModelEvaluator(model_save_path, args.num_eval_games)
        results = evaluator.evaluate_model()
        
        # Generate and print report
        report = evaluator.generate_evaluation_report(results)
        print("\n" + report)
        
        # Save evaluation results
        eval_output_path = os.path.join(args.output_dir, "evaluation_results.json")
        evaluator.save_evaluation_results(results, eval_output_path)
        logger.info(f"Evaluation results saved to {eval_output_path}")


if __name__ == "__main__":
    main() 