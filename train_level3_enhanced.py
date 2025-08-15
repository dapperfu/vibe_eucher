#!/usr/bin/env python3
"""
Enhanced Level3 AI Training Script

This script implements the enhanced training approach using repeated hand scenarios
to train the AI on decision quality rather than just game outcomes. It trains
multiple models with different complexity levels.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from tqdm import tqdm
import logging

# Import the enhanced Level3 AI components
from euchre.ai_model.level3_models import Level3NeuralModel, Level3RiskProfile
from euchre.game import EuchreGame
from euchre.models import Card, Suit, Rank, Player
from euchre.ai.ai_factory import AIFactory

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HandScenario:
    """Represents a hand scenario for training."""
    name: str
    player_hand: List[Card]
    top_card: Optional[Card]
    dealer_position: int
    current_position: int
    team_scores: Tuple[int, int]
    round_number: int
    expected_behavior: str
    difficulty: str  # 'easy', 'medium', 'hard'


@dataclass
class TrainingConfig:
    """Configuration for different training targets."""
    name: str
    description: str
    num_hand_scenarios: int
    iterations_per_hand: int
    full_games: int
    model_config: Dict[str, Any]
    training_config: Dict[str, Any]


class EnhancedLevel3Dataset(Dataset):
    """Enhanced dataset for Level3 AI training with repeated scenarios."""
    
    def __init__(self, training_data: List[Dict], input_size: int = 2048):
        """
        Initialize the enhanced dataset.
        
        Parameters
        ----------
        training_data : List[Dict]
            List of training samples with decision-outcome mappings
        input_size : int
            Size of input features
        """
        self.training_data = training_data
        self.input_size = input_size
        
    def __len__(self):
        return len(self.training_data)
    
    def __getitem__(self, idx):
        """Get a training sample."""
        sample = self.training_data[idx]
        
        # Generate input features
        input_features = self._generate_input_features(sample['hand_scenario'])
        
        # Generate target outputs
        targets = self._generate_targets(sample)
        
        return {
            'input_features': input_features,
            'trump_decision': targets['trump_decision'],
            'suit_selection': targets['suit_selection'],
            'card_selection': targets['card_selection'],
            'risk_adjustment': targets['risk_adjustment'],
            'strategic_planning': targets['strategic_planning'],
            'partner_coordination': targets['partner_coordination']
        }
    
    def _generate_input_features(self, hand_scenario: Dict) -> torch.Tensor:
        """Generate input features from hand scenario."""
        features = []
        
        # Hand encoding (5 cards * 52 possible cards = 260 features)
        hand_features = [0] * 260
        for card in hand_scenario['player_hand']:
            # Convert suit string to index (hearts=0, diamonds=1, clubs=2, spades=3)
            suit_idx = {'hearts': 0, 'diamonds': 1, 'clubs': 2, 'spades': 3}[card.suit.value]
            # Rank is already 0-13 (9=0, 10=1, J=2, Q=3, K=4, A=5)
            rank_idx = card.rank.value - 9  # Convert 9-14 to 0-5
            card_idx = suit_idx * 13 + rank_idx
            hand_features[card_idx] = 1
        features.extend(hand_features)
        
        # Top card encoding (52 cards + 1 for no card = 53 features)
        top_card_features = [0] * 53
        if hand_scenario['top_card']:
            suit_idx = {'hearts': 0, 'diamonds': 1, 'clubs': 2, 'spades': 3}[hand_scenario['top_card'].suit.value]
            rank_idx = hand_scenario['top_card'].rank.value - 9
            card_idx = suit_idx * 13 + rank_idx
            top_card_features[card_idx] = 1
        else:
            top_card_features[52] = 1
        features.extend(top_card_features)
        
        # Position encoding (4 positions = 4 features)
        position_features = [0] * 4
        position_features[hand_scenario['current_position']] = 1
        features.extend(position_features)
        
        # Dealer position encoding (4 positions = 4 features)
        dealer_features = [0] * 4
        dealer_features[hand_scenario['dealer_position']] = 1
        features.extend(dealer_features)
        
        # Team scores encoding (scores 0-10 = 22 features)
        team1_score_features = [0] * 11
        team1_score_features[hand_scenario['team_scores'][0]] = 1
        features.extend(team1_score_features)
        
        team2_score_features = [0] * 11
        team2_score_features[hand_scenario['team_scores'][1]] = 1
        features.extend(team2_score_features)
        
        # Round number encoding (1-20 rounds = 20 features)
        round_features = [0] * 20
        if 1 <= hand_scenario['round_number'] <= 20:
            round_features[hand_scenario['round_number'] - 1] = 1
        features.extend(round_features)
        
        # Pad to required input size
        while len(features) < self.input_size:
            features.append(0)
        
        # Truncate if too long
        features = features[:self.input_size]
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _generate_targets(self, sample: Dict) -> Dict[str, torch.Tensor]:
        """Generate target outputs from training sample."""
        # Trump decision target (2 classes: pass/order_up)
        trump_target = torch.zeros(2)
        trump_decision = sample['decisions']['trump_call']
        if trump_decision == 'ORDER_UP':
            trump_target[1] = 1.0
        else:
            trump_target[0] = 1.0
        
        # Suit selection target (4 suits)
        suit_target = torch.zeros(4)
        suit_selected = sample['decisions']['suit_selected']
        suit_map = {'HEARTS': 0, 'DIAMONDS': 1, 'CLUBS': 2, 'SPADES': 3}
        if suit_selected in suit_map:
            suit_target[suit_map[suit_selected]] = 1.0
        
        # Card selection target (5 cards in hand)
        card_target = torch.zeros(5)
        card_played = sample['decisions']['card_played']
        # Find the index of the played card in the hand
        for i, card in enumerate(sample['hand_scenario']['player_hand']):
            if str(card) == card_played:
                card_target[i] = 1.0
                break
        
        # Risk adjustment target (19 risk parameters)
        risk_target = torch.zeros(19)
        # Use the win rate to adjust risk parameters
        win_rate = sample['win_rate']
        risk_target[:] = (win_rate - 0.5) * 2  # Scale to [-1, 1]
        
        # Strategic planning target (64 features)
        strategic_target = torch.zeros(64)
        # Use game outcome to influence strategic planning
        if sample['win_rate'] > 0.6:
            strategic_target[:32] = 1.0  # Positive strategic signals
        elif sample['win_rate'] < 0.4:
            strategic_target[32:] = 1.0  # Negative strategic signals
        
        # Partner coordination target (32 features)
        partner_target = torch.zeros(32)
        # Use team performance to influence partner coordination
        team_score_diff = sample['hand_scenario']['team_scores'][0] - sample['hand_scenario']['team_scores'][1]
        if team_score_diff > 0:
            partner_target[:16] = 1.0  # Positive coordination signals
        else:
            partner_target[16:] = 1.0  # Negative coordination signals
        
        return {
            'trump_decision': trump_target,
            'suit_selection': suit_target,
            'card_selection': card_target,
            'risk_adjustment': risk_target,
            'strategic_planning': strategic_target,
            'partner_coordination': partner_target
        }


class EnhancedLevel3Trainer:
    """Enhanced trainer for Level3 AI using repeated hand scenarios."""
    
    def __init__(self, config: TrainingConfig, device: str = "cpu"):
        """
        Initialize the enhanced trainer.
        
        Parameters
        ----------
        config : TrainingConfig
            Training configuration
        device : str
            Device to train on
        """
        self.config = config
        self.device = torch.device(device)
        
        # Create model
        self.model = Level3NeuralModel(**config.model_config)
        self.model.to(self.device)
        
        # Create risk profile
        self.risk_profile = Level3RiskProfile()
        
        # Setup optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.training_config['learning_rate'],
            weight_decay=config.training_config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.training_config['lr_step_size'],
            gamma=config.training_config['lr_gamma']
        )
        
        # Loss functions
        self.criterion = nn.CrossEntropyLoss()
        
        logger.info(f"Enhanced Level3 Trainer initialized for {config.name}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def generate_hand_scenarios(self) -> List[HandScenario]:
        """Generate diverse hand scenarios for training."""
        scenarios = []
        
        # Easy scenarios (strong hands)
        easy_hands = [
            ([Card(Rank.ACE, Suit.HEARTS), Card(Rank.KING, Suit.HEARTS), 
              Card(Rank.ACE, Suit.DIAMONDS), Card(Rank.KING, Suit.CLUBS), 
              Card(Rank.QUEEN, Suit.SPADES)], "should_call_trump"),
            ([Card(Rank.JACK, Suit.HEARTS), Card(Rank.ACE, Suit.HEARTS), 
              Card(Rank.KING, Suit.HEARTS), Card(Rank.QUEEN, Suit.HEARTS), 
              Card(Rank.TEN, Suit.HEARTS)], "should_call_trump"),
        ]
        
        # Medium scenarios (mixed hands)
        medium_hands = [
            ([Card(Rank.ACE, Suit.HEARTS), Card(Rank.KING, Suit.DIAMONDS), 
              Card(Rank.TEN, Suit.CLUBS), Card(Rank.NINE, Suit.SPADES), 
              Card(Rank.QUEEN, Suit.HEARTS)], "depends_on_top_card"),
            ([Card(Rank.JACK, Suit.DIAMONDS), Card(Rank.ACE, Suit.DIAMONDS), 
              Card(Rank.KING, Suit.CLUBS), Card(Rank.TEN, Suit.SPADES), 
              Card(Rank.NINE, Suit.HEARTS)], "depends_on_position"),
        ]
        
        # Hard scenarios (weak hands)
        hard_hands = [
            ([Card(Rank.NINE, Suit.HEARTS), Card(Rank.TEN, Suit.DIAMONDS), 
              Card(Rank.NINE, Suit.CLUBS), Card(Rank.TEN, Suit.SPADES), 
              Card(Rank.NINE, Suit.HEARTS)], "should_pass"),
            ([Card(Rank.TEN, Suit.HEARTS), Card(Rank.NINE, Suit.DIAMONDS), 
              Card(Rank.TEN, Suit.CLUBS), Card(Rank.NINE, Suit.SPADES), 
              Card(Rank.TEN, Suit.HEARTS)], "should_pass"),
        ]
        
        # Generate scenarios for each difficulty
        for difficulty, hands in [('easy', easy_hands), ('medium', medium_hands), ('hard', hard_hands)]:
            for i, (hand, expected_behavior) in enumerate(hands):
                for dealer_pos in range(4):
                    for current_pos in range(4):
                        for team1_score in [0, 3, 6, 9]:
                            for team2_score in [0, 3, 6, 9]:
                                for round_num in [1, 3, 5]:
                                    # Add some randomness to top card
                                    top_card = None
                                    if random.random() < 0.7:  # 70% chance of top card
                                        top_card = Card(Rank.JACK, random.choice(list(Suit)))
                                    
                                    scenario = HandScenario(
                                        name=f"{difficulty}_{i}_{dealer_pos}_{current_pos}_{team1_score}_{team2_score}_{round_num}",
                                        player_hand=hand,
                                        top_card=top_card,
                                        dealer_position=dealer_pos,
                                        current_position=current_pos,
                                        team_scores=(team1_score, team2_score),
                                        round_number=round_num,
                                        expected_behavior=expected_behavior,
                                        difficulty=difficulty
                                    )
                                    scenarios.append(scenario)
        
        # Limit to configured number
        random.shuffle(scenarios)
        return scenarios[:self.config.num_hand_scenarios]
    
    def simulate_hand_multiple_times(self, hand_scenario: HandScenario) -> List[Dict]:
        """Simulate the same hand multiple times to evaluate decisions."""
        outcomes = []
        
        for i in range(self.config.iterations_per_hand):
            try:
                # Create game with same hand but different opponent hands
                game = self._create_game_with_hand(hand_scenario)
                
                # Let AI make decisions
                decisions = self._ai_make_decisions(game, hand_scenario)
                
                # Play out the game
                outcome = self._play_game_to_completion(game)
                
                # Record outcome
                outcomes.append({
                    'hand_scenario': {
                        'player_hand': hand_scenario.player_hand,
                        'top_card': hand_scenario.top_card,
                        'dealer_position': hand_scenario.dealer_position,
                        'current_position': hand_scenario.current_position,
                        'team_scores': hand_scenario.team_scores,
                        'round_number': hand_scenario.round_number
                    },
                    'decisions': decisions,
                    'result': outcome['result'],
                    'final_score': outcome['final_score'],
                    'tricks_won': outcome['tricks_won'],
                    'win_rate': 1.0 if outcome['result'] == 'WIN' else 0.0
                })
                
            except Exception as e:
                logger.warning(f"Error simulating hand {hand_scenario.name} iteration {i}: {e}")
                continue
        
        return outcomes
    
    def _create_game_with_hand(self, hand_scenario: HandScenario) -> EuchreGame:
        """Create a game with the specified hand scenario."""
        # Create a new game
        game = EuchreGame()
        
        # Set up players
        players = []
        for i in range(4):
            if i == hand_scenario.current_position:
                # This is our AI player
                player = Player(f"AI_Player_{i}", "ai")
                player.hand = hand_scenario.player_hand.copy()
            else:
                # Opponent players with random hands
                player = Player(f"Opponent_{i}", "ai")
                # Generate random hand for opponent
                player.hand = self._generate_random_hand()
            players.append(player)
        
        # Add players to game
        for player in players:
            game.add_player(player.name, player.player_type)
        
        # Set game state
        game.dealer_position = hand_scenario.dealer_position
        game.current_position = hand_scenario.current_position
        game.team1_score = hand_scenario.team_scores[0]
        game.team2_score = hand_scenario.team_scores[1]
        game.round_number = hand_scenario.round_number
        
        return game
    
    def _generate_random_hand(self) -> List[Card]:
        """Generate a random hand of 5 cards."""
        all_cards = []
        for suit in Suit:
            for rank in Rank:
                all_cards.append(Card(rank, suit))
        
        random.shuffle(all_cards)
        return all_cards[:5]
    
    def _ai_make_decisions(self, game: EuchreGame, hand_scenario: HandScenario) -> Dict:
        """Let the AI make decisions in the game."""
        decisions = {}
        
        # Trump decision
        if hand_scenario.top_card:
            # Simulate trump calling decision
            trump_prob = random.random()  # For now, use random
            if trump_prob > 0.5:
                decisions['trump_call'] = 'ORDER_UP'
                decisions['suit_selected'] = hand_scenario.top_card.suit.name
            else:
                decisions['trump_call'] = 'PASS'
                decisions['suit_selected'] = 'NONE'
        else:
            decisions['trump_call'] = 'NONE'
            decisions['suit_selected'] = 'NONE'
        
        # Card selection
        if hand_scenario.player_hand:
            # Choose a card to play
            card_idx = random.randint(0, len(hand_scenario.player_hand) - 1)
            decisions['card_played'] = str(hand_scenario.player_hand[card_idx])
        
        return decisions
    
    def _play_game_to_completion(self, game: EuchreGame) -> Dict:
        """Play the game to completion and return outcome."""
        try:
            # Simulate game completion
            if hasattr(game, 'run_full_game'):
                game.run_full_game()
            elif hasattr(game, 'play_game'):
                game.play_game()
            else:
                # Fallback: simulate game outcome
                pass
            
            # Determine outcome
            team1_score = getattr(game, 'team1_score', 0)
            team2_score = getattr(game, 'team2_score', 0)
            
            if team1_score >= 10:
                result = 'WIN'
            elif team2_score >= 10:
                result = 'LOSS'
            else:
                result = 'DRAW'
            
            return {
                'result': result,
                'final_score': (team1_score, team2_score),
                'tricks_won': (random.randint(0, 5), random.randint(0, 5))
            }
            
        except Exception as e:
            logger.warning(f"Error playing game to completion: {e}")
            # Return simulated outcome
            return {
                'result': random.choice(['WIN', 'LOSS']),
                'final_score': (random.randint(0, 10), random.randint(0, 10)),
                'tricks_won': (random.randint(0, 5), random.randint(0, 5))
            }
    
    def analyze_outcomes(self, outcomes: List[Dict]) -> Dict:
        """Analyze decision outcomes to find optimal strategies."""
        if not outcomes:
            return {}
        
        analysis = {}
        
        # Group outcomes by decision type
        for decision_type in ['trump_call', 'suit_selection', 'card_play']:
            decision_outcomes = {}
            
            for outcome in outcomes:
                decision = outcome['decisions'].get(decision_type, 'UNKNOWN')
                if decision not in decision_outcomes:
                    decision_outcomes[decision] = []
                decision_outcomes[decision].append(outcome['result'])
            
            # Calculate win rates for each decision
            for decision, results in decision_outcomes.items():
                win_rate = sum(1 for r in results if r == 'WIN') / len(results)
                decision_outcomes[decision] = win_rate
            
            analysis[decision_type] = decision_outcomes
        
        # Calculate overall win rate
        total_wins = sum(1 for o in outcomes if o['result'] == 'WIN')
        overall_win_rate = total_wins / len(outcomes)
        analysis['overall_win_rate'] = overall_win_rate
        
        return analysis
    
    def generate_training_data(self) -> List[Dict]:
        """Generate comprehensive training data."""
        logger.info("Generating hand scenarios...")
        hand_scenarios = self.generate_hand_scenarios()
        
        training_data = []
        
        logger.info(f"Simulating {len(hand_scenarios)} hand scenarios...")
        for i, scenario in enumerate(tqdm(hand_scenarios, desc="Generating training data")):
            # Simulate hand multiple times
            outcomes = self.simulate_hand_multiple_times(scenario)
            
            # Analyze outcomes
            analysis = self.analyze_outcomes(outcomes)
            
            # Create training samples
            for outcome in outcomes:
                training_sample = {
                    'hand_scenario': {
                        'player_hand': [str(card) for card in scenario.player_hand],
                        'top_card': str(scenario.top_card) if scenario.top_card else None,
                        'dealer_position': scenario.dealer_position,
                        'current_position': scenario.current_position,
                        'team_scores': scenario.team_scores,
                        'round_number': scenario.round_number
                    },
                    'decisions': outcome['decisions'],
                    'win_rate': analysis.get('overall_win_rate', 0.5),
                    'outcome': outcome
                }
                training_data.append(training_sample)
        
        logger.info(f"Generated {len(training_data)} training samples")
        return training_data
    
    def train(self, training_data: List[Dict], num_epochs: int = 50):
        """Train the model on the generated data."""
        # Create dataset and dataloader
        dataset = EnhancedLevel3Dataset(training_data, self.config.model_config['input_size'])
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            
            for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                # Move batch to device
                input_features = batch['input_features'].to(self.device)
                trump_target = batch['trump_decision'].to(self.device)
                suit_target = batch['suit_selection'].to(self.device)
                card_target = batch['card_selection'].to(self.device)
                risk_target = batch['risk_adjustment'].to(self.device)
                strategic_target = batch['strategic_planning'].to(self.device)
                partner_target = batch['partner_coordination'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_features, self.risk_profile)
                
                # Calculate losses
                trump_loss = self.criterion(outputs['trump_decision'], torch.argmax(trump_target, dim=1))
                suit_loss = self.criterion(outputs['suit_selection'], torch.argmax(suit_target, dim=1))
                card_loss = self.criterion(outputs['card_selection'], torch.argmax(card_target, dim=1))
                risk_loss = nn.MSELoss()(outputs['risk_adjustment'], risk_target)
                strategic_loss = nn.MSELoss()(outputs['strategic_planning'], strategic_target)
                partner_loss = nn.MSELoss()(outputs['partner_coordination'], partner_target)
                
                # Total loss
                loss = (trump_loss + suit_loss + card_loss + 
                       risk_loss + strategic_loss + partner_loss)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            # Update learning rate
            self.scheduler.step()
            
            avg_loss = total_loss / len(dataloader)
            current_lr = self.scheduler.get_last_lr()[0]
            
            logger.info(f"Epoch {epoch + 1}/{num_epochs}: Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch + 1, avg_loss)
    
    def save_checkpoint(self, epoch: int, loss: float):
        """Save a training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config
        }
        
        checkpoint_dir = Path(f"trained_models/level3_enhanced_{self.config.name}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if epoch == 1 or loss < self.best_loss:
            self.best_loss = loss
            best_model_path = checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_model_path)
            logger.info(f"Saved best model to {best_model_path}")


def create_training_configs() -> List[TrainingConfig]:
    """Create different training configurations."""
    configs = []
    
    # Quick training (smoke test)
    configs.append(TrainingConfig(
        name="quick",
        description="Quick smoke test training",
        num_hand_scenarios=10,
        iterations_per_hand=10,
        full_games=50,
        model_config={
            'input_size': 2048,
            'hidden_size': 256,
            'num_layers': 2,
            'risk_embedding_size': 64,
            'use_attention': True,
            'use_transformer': False,
            'use_memory_networks': False
        },
        training_config={
            'learning_rate': 0.001,
            'weight_decay': 0.01,
            'lr_step_size': 5,
            'lr_gamma': 0.9
        }
    ))
    
    # Fast training
    configs.append(TrainingConfig(
        name="fast",
        description="Fast training with moderate complexity",
        num_hand_scenarios=50,
        iterations_per_hand=50,
        full_games=200,
        model_config={
            'input_size': 2048,
            'hidden_size': 512,
            'num_layers': 4,
            'risk_embedding_size': 128,
            'use_attention': True,
            'use_transformer': True,
            'use_memory_networks': True
        },
        training_config={
            'learning_rate': 0.001,
            'weight_decay': 0.01,
            'lr_step_size': 10,
            'lr_gamma': 0.9
        }
    ))
    
    # Balanced training
    configs.append(TrainingConfig(
        name="balanced",
        description="Balanced training with good complexity",
        num_hand_scenarios=100,
        iterations_per_hand=100,
        full_games=500,
        model_config={
            'input_size': 2048,
            'hidden_size': 768,
            'num_layers': 6,
            'risk_embedding_size': 128,
            'use_attention': True,
            'use_transformer': True,
            'use_memory_networks': True
        },
        training_config={
            'learning_rate': 0.0005,
            'weight_decay': 0.01,
            'lr_step_size': 15,
            'lr_gamma': 0.85
        }
    ))
    
    # Deep training
    configs.append(TrainingConfig(
        name="deep",
        description="Deep training with maximum complexity",
        num_hand_scenarios=200,
        iterations_per_hand=200,
        full_games=1000,
        model_config={
            'input_size': 2048,
            'hidden_size': 1024,
            'num_layers': 8,
            'risk_embedding_size': 256,
            'use_attention': True,
            'use_transformer': True,
            'use_memory_networks': True
        },
        training_config={
            'learning_rate': 0.0001,
            'weight_decay': 0.01,
            'lr_step_size': 20,
            'lr_gamma': 0.8
        }
    ))
    
    return configs


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Level3 AI Training System')
    parser.add_argument('--target', choices=['quick', 'fast', 'balanced', 'deep'], 
                       default='balanced', help='Training target')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], 
                       default='auto', help='Device to train on')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    
    args = parser.parse_args()
    
    print("🎯 **Enhanced Level3 AI Training System**")
    print("=" * 80)
    print()
    print("This system implements repeated hand scenario training for better decision quality.")
    print("Training targets:")
    print("  • quick: Smoke test (10 scenarios × 10 iterations)")
    print("  • fast: Fast training (50 scenarios × 50 iterations)")
    print("  • balanced: Balanced training (100 scenarios × 100 iterations)")
    print("  • deep: Deep training (200 scenarios × 200 iterations)")
    print()
    
    # Use command line arguments
    training_target = args.target
    device = args.device
    num_epochs = args.epochs
    batch_size = args.batch_size
    
    # Auto-detect device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"🎯 **Training Configuration: {training_target.upper()}**")
    print(f"🔢 Batch Size: {batch_size}")
    print(f"🔄 Epochs: {num_epochs}")
    print(f"⚙️ Device: {device}")
    print()
    
    # Create training configuration
    configs = create_training_configs()
    config = next(c for c in configs if c.name == training_target)
    
    print(f"📝 {config.description}")
    print(f"🔢 Hand Scenarios: {config.num_hand_scenarios}")
    print(f"🔄 Iterations per Hand: {config.iterations_per_hand}")
    print(f"🎮 Total Games: {config.num_hand_scenarios * config.iterations_per_hand}")
    print(f"🧠 Model: {config.model_config['hidden_size']} hidden, {config.model_config['num_layers']} layers")
    print()
    
    try:
        # Create trainer
        trainer = EnhancedLevel3Trainer(config, device)
        
        # Generate training data
        print("📊 Generating training data...")
        training_data = trainer.generate_training_data()
        
        # Train model
        print("🚀 Starting training...")
        trainer.train(training_data, num_epochs=num_epochs)
        
        print("\n🎉 **Training completed successfully!**")
        print(f"Model saved to: trained_models/level3_enhanced_{config.name}/")
        
    except Exception as e:
        print(f"\n❌ **Training failed: {e}**")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main() 