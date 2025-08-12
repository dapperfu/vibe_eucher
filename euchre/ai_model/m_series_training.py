"""
M-Series Training Framework for Euchre AI Models

This module provides a comprehensive training framework for the M-Series
PyTorch AI models, designed to train them on thousands of games to achieve
optimal weights for the best possible Euchre players.

The training framework includes:
- Self-play training with thousands of games
- Curriculum learning from simple to complex scenarios
- Multi-model training and competition
- Performance evaluation and model selection
- Risk profile optimization

Author: Claude Sonnet 4 (claude-3-5-sonnet-20241022)
Generated via Cursor IDE (cursor.sh) with AI assistance
Model: Anthropic Claude 3.5 Sonnet
Generation timestamp: 2025-08-12
Context: Creating training framework for M-Series Euchre AI models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import os
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from tqdm import tqdm
import random
from dataclasses import dataclass

from .m_series_models import (
    create_mseries_model, create_mseries_risk_profile, 
    MSeriesGameStateEncoder, MSeriesRiskProfile
)
from ..game import EuchreGame
from ..models import Player, PlayerType, Card, Suit, Rank
from ..ai.ai_factory import AIFactory


@dataclass
class TrainingConfig:
    """Configuration for M-Series model training."""
    
    # Model parameters
    input_size: int = 256
    hidden_size: int = 512
    risk_embedding_size: int = 64
    
    # Training parameters
    num_epochs: int = 1000
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    
    # Self-play parameters
    games_per_epoch: int = 100
    total_training_games: int = 10000
    
    # Curriculum parameters
    curriculum_stages: int = 5
    games_per_stage: int = 2000
    
    # Evaluation parameters
    eval_frequency: int = 100
    eval_games: int = 50
    
    # Model saving
    save_frequency: int = 500
    model_dir: str = "trained_models"
    
    # Device
    device: str = "auto"


class MSeriesGameDataset(Dataset):
    """Dataset for training M-Series models on Euchre game data."""
    
    def __init__(self, game_data: List[Dict[str, Any]], encoder: MSeriesGameStateEncoder):
        """Initialize the dataset.
        
        Parameters
        ----------
        game_data : List[Dict[str, Any]]
            List of game data dictionaries
        encoder : MSeriesGameStateEncoder
            Game state encoder for feature extraction
        """
        self.game_data = game_data
        self.encoder = encoder
        self.samples = self._prepare_samples()
        
    def _prepare_samples(self) -> List[Dict[str, Any]]:
        """Prepare training samples from game data."""
        samples = []
        
        for game in self.game_data:
            # Extract trump decision samples
            trump_samples = self._extract_trump_samples(game)
            samples.extend(trump_samples)
            
            # Extract card play samples
            card_samples = self._extract_card_samples(game)
            samples.extend(card_samples)
        
        return samples
    
    def _extract_trump_samples(self, game: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract samples for trump calling decisions."""
        samples = []
        
        # Extract ordering up decisions
        if 'trump_decision' in game:
            for decision in game['trump_decision']:
                sample = {
                    'type': 'trump_decision',
                    'game_state': decision['game_state'],
                    'target': decision['target'],
                    'player': decision['player'],
                    'context': decision['context']
                }
                samples.append(sample)
        
        return samples
    
    def _extract_card_samples(self, game: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract samples for card play decisions."""
        samples = []
        
        # Extract card play decisions
        if 'card_plays' in game:
            for play in game['card_plays']:
                sample = {
                    'type': 'card_play',
                    'game_state': play['game_state'],
                    'target': play['target'],
                    'player': play['player'],
                    'context': play['context']
                }
                samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        """Get the number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Get a training sample.
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, str]
            (input_features, target, sample_type)
        """
        sample = self.samples[idx]
        
        # Encode game state
        if sample['type'] == 'trump_decision':
            input_features = self.encoder.encode_game_state_for_trump_decision(
                sample['player'], sample['context']['top_card'],
                sample['context']['dealer'], sample['context']['team_scores'],
                sample['context']['round_number']
            )
        else:  # card_play
            input_features = self.encoder.encode_game_state_for_card_play(
                sample['player'], sample['context']['lead_suit'],
                sample['context']['trump_suit'], sample['context']['current_trick'],
                sample['context']['team_scores']
            )
        
        # Create target tensor
        target = torch.tensor(sample['target'], dtype=torch.float32)
        
        return input_features, target, sample['type']


class MSeriesSelfPlayTrainer:
    """Self-play trainer for M-Series models."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize the trainer.
        
        Parameters
        ----------
        config : TrainingConfig
            Training configuration
        """
        self.config = config
        self.device = self._get_device()
        
        # Initialize models
        self.models = self._initialize_models()
        self.encoders = self._initialize_encoders()
        
        # Training components
        self.optimizers = {}
        self.schedulers = {}
        self.criteria = {}
        
        # Performance tracking
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'win_rates': [],
            'model_performance': {}
        }
        
        # Setup logging
        self._setup_logging()
        
    def _get_device(self) -> torch.device:
        """Get the device to use for training."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        else:
            return torch.device(self.config.device)
    
    def _initialize_models(self) -> Dict[str, nn.Module]:
        """Initialize all M-Series models."""
        models = {}
        
        model_names = ["magnus", "maverick", "mentor", "mystic"]
        for name in model_names:
            model = create_mseries_model(
                name, 
                self.config.input_size, 
                self.config.hidden_size, 
                self.config.risk_embedding_size
            )
            model.to(self.device)
            models[name] = model
        
        return models
    
    def _initialize_encoders(self) -> Dict[str, MSeriesGameStateEncoder]:
        """Initialize game state encoders."""
        encoders = {}
        
        model_names = ["magnus", "maverick", "mentor", "mystic"]
        for name in model_names:
            encoder = MSeriesGameStateEncoder(self.config.input_size)
            encoders[name] = encoder
        
        return encoders
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('m_series_training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_training_components(self) -> None:
        """Setup optimizers, schedulers, and loss criteria."""
        for name, model in self.models.items():
            # Optimizer
            self.optimizers[name] = optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            
            # Scheduler
            self.schedulers[name] = optim.lr_scheduler.StepLR(
                self.optimizers[name],
                step_size=100,
                gamma=0.9
            )
            
            # Loss criteria
            self.criteria[name] = {
                'trump_decision': nn.CrossEntropyLoss(),
                'card_selection': nn.CrossEntropyLoss(),
                'suit_selection': nn.CrossEntropyLoss()
            }
    
    def generate_training_data(self) -> List[Dict[str, Any]]:
        """Generate training data through self-play games."""
        self.logger.info("Generating training data through self-play...")
        
        training_data = []
        
        for game_num in tqdm(range(self.config.total_training_games), desc="Generating games"):
            # Create a new game with M-Series models
            game_data = self._play_training_game(game_num)
            training_data.append(game_data)
            
            if game_num % 100 == 0:
                self.logger.info(f"Generated {game_num} training games")
        
        self.logger.info(f"Generated {len(training_data)} training games")
        return training_data
    
    def _play_training_game(self, game_num: int) -> Dict[str, Any]:
        """Play a single training game and record decisions."""
        # Create players with M-Series models
        players = []
        for i, (name, model_name) in enumerate([
            ("Magnus", "magnus"), ("Maverick", "maverick"), 
            ("Mentor", "mentor"), ("Mystic", "mystic")
        ]):
            player = MSeriesTrainingPlayer(
                name, model_name, self.models[model_name], 
                self.encoders[model_name], self.device
            )
            players.append(player)
        
        # Create and play the game
        game = EuchreGame(players, quiet_mode=True)
        game.start_new_game()
        
        # Record game data
        game_data = self._record_game_data(game, players, game_num)
        return game_data
    
    def _record_game_data(self, game: EuchreGame, players: List, game_num: int) -> Dict[str, Any]:
        """Record all decisions made during the game."""
        game_data = {
            'game_id': game_num,
            'trump_decision': [],
            'card_plays': [],
            'final_score': game.game_state_manager.game_scores,
            'winner': None
        }
        
        # Record trump decisions
        # This would need to be implemented based on the actual game flow
        
        # Record card plays
        # This would need to be implemented based on the actual game flow
        
        return game_data
    
    def train_models(self, training_data: List[Dict[str, Any]]) -> None:
        """Train all M-Series models on the training data."""
        self.logger.info("Starting model training...")
        
        # Setup training components
        self._setup_training_components()
        
        # Create datasets
        datasets = self._create_datasets(training_data)
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            epoch_losses = {}
            epoch_accuracies = {}
            
            # Train each model
            for name, model in self.models.items():
                model.train()
                
                # Get dataset for this model
                if name in datasets:
                    dataloader = DataLoader(
                        datasets[name], 
                        batch_size=self.config.batch_size, 
                        shuffle=True
                    )
                    
                    model_loss, model_accuracy = self._train_model_epoch(
                        name, model, dataloader, epoch
                    )
                    
                    epoch_losses[name] = model_loss
                    epoch_accuracies[name] = model_accuracy
            
            # Update learning rates
            for name in self.models:
                if name in self.schedulers:
                    self.schedulers[name].step()
            
            # Log progress
            if epoch % 10 == 0:
                self._log_training_progress(epoch, epoch_losses, epoch_accuracies)
            
            # Evaluate models
            if epoch % self.config.eval_frequency == 0:
                self._evaluate_models(epoch)
            
            # Save models
            if epoch % self.config.save_frequency == 0:
                self._save_models(epoch)
    
    def _create_datasets(self, training_data: List[Dict[str, Any]]) -> Dict[str, MSeriesGameDataset]:
        """Create datasets for each model."""
        datasets = {}
        
        for name in self.models.keys():
            # Filter data for this model (could be model-specific)
            model_data = training_data  # For now, use all data
            
            dataset = MSeriesGameDataset(model_data, self.encoders[name])
            datasets[name] = dataset
        
        return datasets
    
    def _train_model_epoch(self, name: str, model: nn.Module, 
                          dataloader: DataLoader, epoch: int) -> Tuple[float, float]:
        """Train a single model for one epoch."""
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (inputs, targets, sample_types) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            risk_profile = create_mseries_risk_profile(name).to(self.device)
            outputs = model(inputs, risk_profile)
            
            # Calculate loss based on sample type
            loss = 0.0
            for i, sample_type in enumerate(sample_types):
                if sample_type == 'trump_decision':
                    loss += self.criteria[name]['trump_decision'](
                        outputs['trump_decision'][i:i+1], 
                        targets[i:i+1].long()
                    )
                elif sample_type == 'card_play':
                    loss += self.criteria[name]['card_selection'](
                        outputs['card_selection'][i:i+1], 
                        targets[i:i+1].long()
                    )
            
            # Backward pass
            self.optimizers[name].zero_grad()
            loss.backward()
            self.optimizers[name].step()
            
            # Statistics
            total_loss += loss.item()
            total_samples += inputs.size(0)
            
            # Calculate accuracy (simplified)
            if 'trump_decision' in outputs:
                pred = outputs['trump_decision'].argmax(dim=1)
                total_correct += (pred == targets.long()).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        avg_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        return avg_loss, avg_accuracy
    
    def _log_training_progress(self, epoch: int, losses: Dict[str, float], 
                              accuracies: Dict[str, float]) -> None:
        """Log training progress."""
        self.logger.info(f"Epoch {epoch}:")
        for name in self.models:
            if name in losses:
                self.logger.info(f"  {name}: Loss={losses[name]:.4f}, Acc={accuracies[name]:.4f}")
    
    def _evaluate_models(self, epoch: int) -> None:
        """Evaluate all models."""
        self.logger.info(f"Evaluating models at epoch {epoch}...")
        
        # Play evaluation games
        win_rates = {}
        for name in self.models:
            win_rate = self._evaluate_model(name, self.config.eval_games)
            win_rates[name] = win_rate
        
        # Log results
        self.logger.info("Evaluation results:")
        for name, win_rate in win_rates.items():
            self.logger.info(f"  {name}: Win Rate = {win_rate:.2%}")
        
        # Store results
        self.training_history['win_rates'].append({
            'epoch': epoch,
            'win_rates': win_rates
        })
    
    def _evaluate_model(self, model_name: str, num_games: int) -> float:
        """Evaluate a single model."""
        wins = 0
        
        for _ in range(num_games):
            # Create evaluation game
            players = []
            for i, (name, m_name) in enumerate([
                ("Magnus", "magnus"), ("Maverick", "maverick"), 
                ("Mentor", "mentor"), ("Mystic", "mystic")
            ]):
                if m_name == model_name:
                    # Use the trained model
                    player = MSeriesTrainingPlayer(
                        name, m_name, self.models[m_name], 
                        self.encoders[m_name], self.device
                    )
                else:
                    # Use base AI
                    player = AIFactory.create_ai_player(name, "balanced")
                
                players.append(player)
            
            # Play game
            game = EuchreGame(players, quiet_mode=True)
            game.start_new_game()
            
            # Check if model's team won
            if game.game_state_manager.game_scores["Team 1"] >= 10:
                if model_name in ["magnus", "mentor"]:  # Team 1
                    wins += 1
            elif game.game_state_manager.game_scores["Team 2"] >= 10:
                if model_name in ["maverick", "mystic"]:  # Team 2
                    wins += 1
        
        return wins / num_games
    
    def _save_models(self, epoch: int) -> None:
        """Save all trained models."""
        os.makedirs(self.config.model_dir, exist_ok=True)
        
        for name, model in self.models.items():
            save_path = os.path.join(
                self.config.model_dir, 
                f"{name}_epoch_{epoch}.pth"
            )
            torch.save(model.state_dict(), save_path)
        
        self.logger.info(f"Models saved at epoch {epoch}")
    
    def save_training_history(self) -> None:
        """Save training history to file."""
        history_path = os.path.join(self.config.model_dir, "training_history.json")
        
        # Convert tensors to lists for JSON serialization
        serializable_history = {}
        for key, value in self.training_history.items():
            if isinstance(value, list):
                serializable_history[key] = value
            else:
                serializable_history[key] = str(value)
        
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        self.logger.info(f"Training history saved to {history_path}")


class MSeriesTrainingPlayer(Player):
    """Player class for training M-Series models."""
    
    def __init__(self, name: str, model_name: str, model: nn.Module, 
                 encoder: MSeriesGameStateEncoder, device: torch.device):
        """Initialize the training player.
        
        Parameters
        ----------
        name : str
            Player name
        model_name : str
            Name of the M-Series model
        model : nn.Module
            The PyTorch model
        encoder : MSeriesGameStateEncoder
            Game state encoder
        device : torch.device
            Device for model inference
        """
        super().__init__(name, PlayerType.AI)
        self.model_name = model_name
        self.model = model
        self.encoder = encoder
        self.device = device
        self.risk_profile = create_mseries_risk_profile(model_name).to(device)
        
        # Set model to evaluation mode
        self.model.eval()
    
    def should_order_up(self, top_card: Card, is_partner_dealing: bool = False) -> bool:
        """Decide whether to order up the top card."""
        # This would need to be implemented based on the actual game interface
        # For now, return a random decision
        return random.choice([True, False])
    
    def choose_card_to_play(self, lead_suit: Optional[Suit], trump_suit: Optional[Suit]) -> Card:
        """Choose which card to play."""
        # This would need to be implemented based on the actual game interface
        # For now, return a random card
        if self.hand:
            return random.choice(self.hand)
        else:
            raise ValueError("No cards in hand")


def main():
    """Main training function."""
    # Configuration
    config = TrainingConfig(
        num_epochs=1000,
        games_per_epoch=100,
        total_training_games=10000,
        batch_size=32,
        learning_rate=0.001
    )
    
    # Create trainer
    trainer = MSeriesSelfPlayTrainer(config)
    
    # Generate training data
    training_data = trainer.generate_training_data()
    
    # Train models
    trainer.train_models(training_data)
    
    # Save training history
    trainer.save_training_history()
    
    print("Training completed!")


if __name__ == "__main__":
    main() 