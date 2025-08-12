"""Training pipeline for the euchre neural network model."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import logging

from .euchre_nn import EuchreNN as EuchreNeuralNetwork
from .game_state_encoder import GameStateEncoder
from ..models import Card, Suit, Rank, Player, GameState


class EuchreGameDataset(Dataset):
    """Dataset for euchre game data."""
    
    def __init__(self, data_dir: str, encoder: GameStateEncoder) -> None:
        """Initialize the dataset.
        
        Parameters
        ----------
        data_dir : str
            Directory containing game data files
        encoder : GameStateEncoder
            Encoder for converting game states to tensors
        """
        self.data_dir = Path(data_dir)
        self.encoder = encoder
        self.game_files = list(self.data_dir.glob("*.json"))
        self.samples = []
        
        # Load and preprocess all game data
        self._load_game_data()
        
    def _load_game_data(self) -> None:
        """Load and preprocess all game data files."""
        logging.info(f"Loading game data from {len(self.game_files)} files...")
        
        for game_file in self.game_files:
            try:
                with open(game_file, 'r') as f:
                    game_data = json.load(f)
                self._process_game(game_data)
            except Exception as e:
                logging.warning(f"Failed to load {game_file}: {e}")
                
        logging.info(f"Loaded {len(self.samples)} training samples")
        
    def _process_game(self, game_data: Dict[str, Any]) -> None:
        """Process a single game and extract training samples.
        
        Parameters
        ----------
        game_data : Dict[str, Any]
            Game data dictionary
        """
        # Extract game actions and outcomes
        actions = game_data.get('actions', [])
        game_state = game_data.get('game_state', {})
        
        for action in actions:
            sample = self._create_training_sample(action, game_state)
            if sample:
                self.samples.append(sample)
    
    def _create_training_sample(self, 
                               action: Dict[str, Any], 
                               game_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a training sample from an action.
        
        Parameters
        ----------
        action : Dict[str, Any]
            Action data
        game_state : Dict[str, Any]
            Game state data
            
        Returns
        -------
        Optional[Dict[str, Any]]
            Training sample or None if invalid
        """
        action_type = action.get('type')
        
        if action_type == 'order_up':
            return self._create_ordering_sample(action, game_state)
        elif action_type == 'play_card':
            return self._create_card_selection_sample(action, game_state)
        elif action_type == 'call_trump':
            return self._create_trump_selection_sample(action, game_state)
        
        return None
    
    def _create_ordering_sample(self, 
                               action: Dict[str, Any], 
                               game_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create training sample for trump ordering.
        
        Parameters
        ----------
        action : Dict[str, Any]
            Ordering action data
        game_state : Dict[str, Any]
            Game state data
            
        Returns
        -------
        Optional[Dict[str, Any]]
            Training sample
        """
        try:
            # Extract player hand and top card
            player_hand = self._parse_hand(action.get('player_hand', []))
            top_card = self._parse_card(action.get('top_card', {}))
            
            # Create player object
            player = Player("temp", "ai")
            player.hand = player_hand
            
            # Encode game state
            features = self.encoder.encode_game_state_for_ordering(player, top_card)
            
            # Target: 1 if ordered up, 0 if passed
            target = 1 if action.get('ordered_up', False) else 0
            
            return {
                'features': features,
                'target': torch.tensor([target], dtype=torch.long),
                'action_type': 'order_up'
            }
            
        except Exception as e:
            logging.warning(f"Failed to create ordering sample: {e}")
            return None
    
    def _create_card_selection_sample(self, 
                                    action: Dict[str, Any], 
                                    game_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create training sample for card selection.
        
        Parameters
        ----------
        action : Dict[str, Any]
            Card selection action data
        game_state : Dict[str, Any]
            Game state data
            
        Returns
        -------
        Optional[Dict[str, Any]]
            Training sample
        """
        try:
            # Extract player hand, lead suit, trump suit
            player_hand = self._parse_hand(action.get('player_hand', []))
            lead_suit = self._parse_suit(action.get('lead_suit'))
            trump_suit = self._parse_suit(action.get('trump_suit'))
            played_card = self._parse_card(action.get('played_card', {}))
            
            # Create player object
            player = Player("temp", "ai")
            player.hand = player_hand
            
            # Encode game state
            features = self.encoder.encode_game_state_for_card_selection(
                player, lead_suit, trump_suit
            )
            
            # Target: index of played card in hand
            target = self._find_card_index(played_card, player_hand)
            if target is None:
                return None
                
            return {
                'features': features,
                'target': torch.tensor([target], dtype=torch.long),
                'action_type': 'card_selection'
            }
            
        except Exception as e:
            logging.warning(f"Failed to create card selection sample: {e}")
            return None
    
    def _create_trump_selection_sample(self, 
                                     action: Dict[str, Any], 
                                     game_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create training sample for trump selection.
        
        Parameters
        ----------
        action : Dict[str, Any]
            Trump selection action data
        game_state : Dict[str, Any]
            Game state data
            
        Returns
        -------
        Optional[Dict[str, Any]]
            Training sample
        """
        try:
            # Extract dealer hand and called trump
            dealer_hand = self._parse_hand(action.get('dealer_hand', []))
            called_trump = self._parse_suit(action.get('called_trump', {}))
            
            # Encode game state
            features = self.encoder.encode_game_state_for_trump_selection(dealer_hand)
            
            # Target: suit index (0-3)
            target = called_trump.value - 1 if called_trump else 0
            
            return {
                'features': features,
                'target': torch.tensor([target], dtype=torch.long),
                'action_type': 'trump_selection'
            }
            
        except Exception as e:
            logging.warning(f"Failed to create trump selection sample: {e}")
            return None
    
    def _parse_hand(self, hand_data: List[Dict[str, Any]]) -> List[Card]:
        """Parse hand data into Card objects.
        
        Parameters
        ----------
        hand_data : List[Dict[str, Any]]
            List of card data dictionaries
            
        Returns
        -------
        List[Card]
            List of Card objects
        """
        cards = []
        for card_data in hand_data:
            card = self._parse_card(card_data)
            if card:
                cards.append(card)
        return cards
    
    def _parse_card(self, card_data: Dict[str, Any]) -> Optional[Card]:
        """Parse card data into Card object.
        
        Parameters
        ----------
        card_data : Dict[str, Any]
            Card data dictionary
            
        Returns
        -------
        Optional[Card]
            Card object or None if invalid
        """
        try:
            suit = Suit(card_data.get('suit', 'hearts'))
            rank = Rank(card_data.get('rank', 9))
            is_trump = card_data.get('is_trump', False)
            
            return Card(rank=rank, suit=suit, is_trump=is_trump)
        except Exception:
            return None
    
    def _parse_suit(self, suit_data: Any) -> Optional[Suit]:
        """Parse suit data into Suit enum.
        
        Parameters
        ----------
        suit_data : Any
            Suit data
            
        Returns
        -------
        Optional[Suit]
            Suit enum or None if invalid
        """
        if not suit_data:
            return None
        try:
            return Suit(suit_data)
        except Exception:
            return None
    
    def _find_card_index(self, card: Card, hand: List[Card]) -> Optional[int]:
        """Find index of card in hand.
        
        Parameters
        ----------
        card : Card
            Card to find
        hand : List[Card]
            Hand to search in
            
        Returns
        -------
        Optional[int]
            Index of card or None if not found
        """
        for i, hand_card in enumerate(hand):
            if (hand_card.rank == card.rank and 
                hand_card.suit == card.suit and 
                hand_card.is_trump == card.is_trump):
                return i
        return None
    
    def __len__(self) -> int:
        """Get number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get sample by index."""
        return self.samples[idx]


class TrainingPipeline:
    """Pipeline for training the euchre neural network."""
    
    def __init__(self, 
                 model: EuchreNeuralNetwork,
                 data_dir: str,
                 batch_size: int = 32,
                 learning_rate: float = 0.001) -> None:
        """Initialize the training pipeline.
        
        Parameters
        ----------
        model : EuchreNeuralNetwork
            Neural network model to train
        data_dir : str
            Directory containing training data
        batch_size : int
            Batch size for training
        learning_rate : float
            Learning rate for optimizer
        """
        self.model = model
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        self.encoder = GameStateEncoder(input_size=model.input_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def train(self, 
              num_epochs: int = 100,
              validation_split: float = 0.2,
              save_path: str = "models/euchre_model.pth") -> Dict[str, List[float]]:
        """Train the model.
        
        Parameters
        ----------
        num_epochs : int
            Number of training epochs
        validation_split : float
            Fraction of data to use for validation
        save_path : str
            Path to save the trained model
            
        Returns
        -------
        Dict[str, List[float]]
            Training history with loss and accuracy
        """
        # Create datasets
        full_dataset = EuchreGameDataset(self.data_dir, self.encoder)
        
        # Split into train/validation
        train_size = int((1 - validation_split) * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        self.logger.info(f"Starting training for {num_epochs} epochs...")
        self.logger.info(f"Training samples: {len(train_dataset)}")
        self.logger.info(f"Validation samples: {len(val_dataset)}")
        
        for epoch in range(num_epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            # Record history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Log progress
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
            
            # Save best model
            if val_acc == max(history['val_acc']):
                self._save_model(save_path)
                self.logger.info(f"New best model saved to {save_path}")
        
        return history
    
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch.
        
        Parameters
        ----------
        train_loader : DataLoader
            Training data loader
            
        Returns
        -------
        Tuple[float, float]
            Average loss and accuracy for the epoch
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch in train_loader:
            features = batch['features'].to(self.device)
            targets = batch['target'].squeeze().to(self.device)
            action_types = batch['action_type']
            
            # Forward pass
            outputs = self.model(features)
            
            # Calculate loss for each action type
            batch_loss = 0.0
            batch_correct = 0
            
            for i, action_type in enumerate(action_types):
                if action_type == 'order_up':
                    loss = self.criterion(outputs['order_up'][i:i+1], targets[i:i+1])
                    pred = torch.argmax(outputs['order_up'][i:i+1], dim=1)
                elif action_type == 'card_selection':
                    loss = self.criterion(outputs['card_selection'][i:i+1], targets[i:i+1])
                    pred = torch.argmax(outputs['card_selection'][i:i+1], dim=1)
                elif action_type == 'trump_selection':
                    loss = self.criterion(outputs['trump_selection'][i:i+1], targets[i:i+1])
                    pred = torch.argmax(outputs['trump_selection'][i:i+1], dim=1)
                else:
                    continue
                
                batch_loss += loss
                batch_correct += (pred == targets[i:i+1]).sum().item()
            
            # Backward pass
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += batch_loss.item()
            total_correct += batch_correct
            total_samples += len(features)
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc
    
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch.
        
        Parameters
        ----------
        val_loader : DataLoader
            Validation data loader
            
        Returns
        -------
        Tuple[float, float]
            Average loss and accuracy for the epoch
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(self.device)
                targets = batch['target'].squeeze().to(self.device)
                action_types = batch['action_type']
                
                # Forward pass
                outputs = self.model(features)
                
                # Calculate loss for each action type
                batch_loss = 0.0
                batch_correct = 0
                
                for i, action_type in enumerate(action_types):
                    if action_type == 'order_up':
                        loss = self.criterion(outputs['order_up'][i:i+1], targets[i:i+1])
                        pred = torch.argmax(outputs['order_up'][i:i+1], dim=1)
                    elif action_type == 'card_selection':
                        loss = self.criterion(outputs['card_selection'][i:i+1], targets[i:i+1])
                        pred = torch.argmax(outputs['card_selection'][i:i+1], dim=1)
                    elif action_type == 'trump_selection':
                        loss = self.criterion(outputs['trump_selection'][i:i+1], targets[i:i+1])
                        pred = torch.argmax(outputs['trump_selection'][i:i+1], dim=1)
                    else:
                        continue
                    
                    batch_loss += loss
                    batch_correct += (pred == targets[i:i+1]).sum().item()
                
                # Update metrics
                total_loss += batch_loss.item()
                total_correct += batch_correct
                total_samples += len(features)
        
        avg_loss = total_loss / len(val_loader)
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc
    
    def _save_model(self, save_path: str) -> None:
        """Save the trained model.
        
        Parameters
        ----------
        save_path : str
            Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save model
        self.model.save_model(save_path)
        self.logger.info(f"Model saved to {save_path}")
    
    def generate_sample_data(self, num_games: int = 1000) -> None:
        """Generate sample training data for testing.
        
        Parameters
        ----------
        num_games : int
            Number of sample games to generate
        """
        self.logger.info(f"Generating {num_games} sample games...")
        
        # Create data directory
        os.makedirs(self.data_dir, exist_ok=True)
        
        for i in range(num_games):
            game_data = self._generate_sample_game()
            
            # Save game data
            filename = os.path.join(self.data_dir, f"game_{i:06d}.json")
            with open(filename, 'w') as f:
                json.dump(game_data, f, indent=2)
        
        self.logger.info(f"Generated {num_games} sample games in {self.data_dir}")
    
    def _generate_sample_game(self) -> Dict[str, Any]:
        """Generate a sample game for training data.
        
        Returns
        -------
        Dict[str, Any]
            Sample game data
        """
        # This is a simplified sample generator
        # In practice, you'd want more sophisticated game generation
        
        game_data = {
            'game_id': f"sample_{np.random.randint(1000000)}",
            'actions': [],
            'game_state': {}
        }
        
        # Generate some sample actions
        for action_num in range(20):  # 20 actions per game
            action_type = np.random.choice(['order_up', 'play_card', 'call_trump'])
            
            if action_type == 'order_up':
                action = {
                    'type': 'order_up',
                    'player_hand': self._generate_sample_hand(),
                    'top_card': self._generate_sample_card(),
                    'ordered_up': np.random.choice([True, False], p=[0.3, 0.7])
                }
            elif action_type == 'play_card':
                action = {
                    'type': 'play_card',
                    'player_hand': self._generate_sample_hand(),
                    'lead_suit': np.random.choice(['hearts', 'diamonds', 'clubs', 'spades']),
                    'trump_suit': np.random.choice(['hearts', 'diamonds', 'clubs', 'spades']),
                    'played_card': self._generate_sample_card()
                }
            else:  # call_trump
                action = {
                    'type': 'call_trump',
                    'dealer_hand': self._generate_sample_hand(),
                    'called_trump': np.random.choice(['hearts', 'diamonds', 'clubs', 'spades'])
                }
            
            game_data['actions'].append(action)
        
        return game_data
    
    def _generate_sample_hand(self) -> List[Dict[str, Any]]:
        """Generate a sample hand of 5 cards.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of card dictionaries
        """
        suits = ['hearts', 'diamonds', 'clubs', 'spades']
        ranks = [9, 10, 11, 12, 13, 14]  # 9, 10, J, Q, K, A
        
        hand = []
        for _ in range(5):
            card = {
                'suit': np.random.choice(suits),
                'rank': np.random.choice(ranks),
                'is_trump': np.random.choice([True, False], p=[0.2, 0.8])
            }
            hand.append(card)
        
        return hand
    
    def _generate_sample_card(self) -> Dict[str, Any]:
        """Generate a sample card.
        
        Returns
        -------
        Dict[str, Any]
            Card dictionary
        """
        suits = ['hearts', 'diamonds', 'clubs', 'spades']
        ranks = [9, 10, 11, 12, 13, 14]
        
        return {
            'suit': np.random.choice(suits),
            'rank': np.random.choice(ranks),
            'is_trump': np.random.choice([True, False], p=[0.2, 0.8])
        } 