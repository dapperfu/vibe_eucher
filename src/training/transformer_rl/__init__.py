"""Transformer RL training system."""

from src.training.transformer_rl.curriculum_trainer import CurriculumTrainer, TrainingStage
from src.training.transformer_rl.train_transformer_rl import train_transformer_rl, TrainingEpisode
from src.training.transformer_rl.evaluation import EvaluationMetrics, evaluate_agent, tournament_evaluation

__all__ = [
    "CurriculumTrainer",
    "TrainingStage",
    "train_transformer_rl",
    "TrainingEpisode",
    "EvaluationMetrics",
    "evaluate_agent",
    "tournament_evaluation",
]

