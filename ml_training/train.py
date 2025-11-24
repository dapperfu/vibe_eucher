"""Training pipeline for ML models."""

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from src.cards import Card, Rank, Suit
from src.ml_config import MLConfig
from src.ml_features import GameStateEncoder
from src.ml_model import EuchreMLModel


class EuchreDataset(Dataset):
    """PyTorch dataset for Euchre training examples."""

    def __init__(self, examples: List[dict], encoder: GameStateEncoder, action_type: str) -> None:
        """
        Initialize the dataset.

        Parameters
        ----------
        examples : List[dict]
            List of training examples (from JSON).
        encoder : GameStateEncoder
            Feature encoder.
        action_type : str
            Type of action: "order_up", "call_trump", "play_card", "discard".
        """
        self.examples = [ex for ex in examples if ex["action_type"] == action_type]
        self.encoder = encoder
        self.action_type = action_type

    def __len__(self) -> int:
        """
        Get dataset size.

        Returns
        -------
        int
            Number of examples.
        """
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Get a training example.

        Parameters
        ----------
        idx : int
            Example index.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
            (features, target, outcome).
        """
        ex = self.examples[idx]

        # Parse cards from strings (simplified - would need proper parsing)
        hand = self._parse_cards(ex["hand"])
        turned_card = self._parse_card(ex["turned_card"]) if ex["turned_card"] else None
        trump_suit = Suit(ex["trump_suit"]) if ex["trump_suit"] else None
        led_suit = Suit(ex["led_suit"]) if ex["led_suit"] else None
        trick_cards = self._parse_cards(ex["trick_cards"])

        # Encode features
        features = self.encoder.encode_game_state(
            hand=hand,
            turned_card=turned_card,
            trump_suit=trump_suit,
            led_suit=led_suit,
            trick_cards=trick_cards,
            player_id=ex["player_id"],
            dealer_id=ex["dealer_id"],
            team=ex["team"],
            trick_number=ex["trick_number"],
            tricks_won_team0=ex["tricks_won_team0"],
            tricks_won_team1=ex["tricks_won_team1"],
        )

        # Encode target
        target = self._encode_target(ex["action_value"])

        # Outcome (optional)
        outcome = torch.tensor([ex["outcome"]], dtype=torch.float32) if ex.get("outcome") is not None else None

        return features, target, outcome

    def _parse_card(self, card_str: Optional[str]) -> Optional[Card]:
        """
        Parse a card from string representation.

        Parameters
        ----------
        card_str : Optional[str]
            Card string representation.

        Returns
        -------
        Optional[Card]
            The card, or None.
        """
        if not card_str:
            return None

        # Simplified parsing - assumes format like "K♥" or "A♠"
        # Full implementation would need proper parsing
        try:
            rank_map = {"9": Rank.NINE, "10": Rank.TEN, "J": Rank.JACK, "Q": Rank.QUEEN, "K": Rank.KING, "A": Rank.ACE}
            suit_map = {"♥": Suit.HEARTS, "♦": Suit.DIAMONDS, "♣": Suit.CLUBS, "♠": Suit.SPADES}

            if len(card_str) >= 2:
                rank_str = card_str[:-1]
                suit_str = card_str[-1]
                return Card(suit_map[suit_str], rank_map[rank_str])
        except Exception:
            pass

        return None

    def _parse_cards(self, card_strs: List[str]) -> List[Card]:
        """
        Parse a list of cards.

        Parameters
        ----------
        card_strs : List[str]
            List of card string representations.

        Returns
        -------
        List[Card]
            List of cards.
        """
        return [card for card in [self._parse_card(s) for s in card_strs] if card is not None]

    def _encode_target(self, action_value: str) -> torch.Tensor:
        """
        Encode the target action.

        Parameters
        ----------
        action_value : str
            Action value string.

        Returns
        -------
        torch.Tensor
            Encoded target.
        """
        if self.action_type == "order_up":
            # Binary classification
            return torch.tensor([1.0 if action_value == "True" else 0.0], dtype=torch.float32)

        elif self.action_type == "call_trump":
            # Multi-class: [pass, HEARTS, DIAMONDS, CLUBS, SPADES]
            target = torch.zeros(5, dtype=torch.float32)
            if action_value == "pass":
                target[0] = 1.0
            else:
                suit_map = {"Hearts": 1, "Diamonds": 2, "Clubs": 3, "Spades": 4}
                if action_value in suit_map:
                    target[suit_map[action_value]] = 1.0
            return target

        elif self.action_type == "play_card" or self.action_type == "discard":
            # Card selection: one-hot over 24 cards
            card = self._parse_card(action_value)
            target = torch.zeros(24, dtype=torch.float32)
            if card and card in self.encoder.card_to_index:
                target[self.encoder.card_to_index[card]] = 1.0
            return target

        return torch.tensor([0.0])


def train_model(
    model: EuchreMLModel,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    config: MLConfig,
    num_epochs: int,
    model_name: str,
) -> None:
    """
    Train a model.

    Parameters
    ----------
    model : EuchreMLModel
        The model to train.
    train_loader : DataLoader
        Training data loader.
    val_loader : Optional[DataLoader]
        Validation data loader.
    config : MLConfig
        ML configuration.
    num_epochs : int
        Number of training epochs.
    model_name : str
        Name for saving the model.
    """
    device = model.device

    # Select which network to train based on model_name
    if "trump" in model_name:
        network = model.trump_net
        criterion_order = nn.BCEWithLogitsLoss()
        criterion_call = nn.CrossEntropyLoss()
    elif "card" in model_name:
        network = model.card_play_net
        criterion = nn.CrossEntropyLoss()
    elif "discard" in model_name:
        network = model.discard_net
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    optimizer = optim.Adam(network.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    network.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        for features, target, outcome in train_loader:
            features = features.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            if "trump" in model_name:
                order_logits, call_logits = network(features)
                # Use appropriate loss based on target shape
                if target.shape[1] == 1:
                    # Order up
                    loss = criterion_order(order_logits, target)
                else:
                    # Call trump
                    loss = criterion_call(call_logits, target.argmax(dim=1))
            else:
                output = network(features)
                loss = criterion(output, target.argmax(dim=1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Validation
        if val_loader is not None:
            network.eval()
            val_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                for features, target, outcome in val_loader:
                    features = features.to(device)
                    target = target.to(device)

                    if "trump" in model_name:
                        order_logits, call_logits = network(features)
                        if target.shape[1] == 1:
                            loss = criterion_order(order_logits, target)
                        else:
                            loss = criterion_call(call_logits, target.argmax(dim=1))
                    else:
                        output = network(features)
                        loss = criterion(output, target.argmax(dim=1))

                    val_loss += loss.item()
                    val_batches += 1

            avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
            print(f"  Validation Loss: {avg_val_loss:.4f}")
            network.train()

    # Save model
    model_path = config.get_model_path(f"{model_name}.pth")
    if "trump" in model_name:
        torch.save(network.state_dict(), model_path)
    elif "card" in model_name:
        torch.save(network.state_dict(), model_path)
    elif "discard" in model_name:
        torch.save(network.state_dict(), model_path)

    print(f"Model saved to {model_path}")


def main() -> None:
    """Main training function."""
    config = MLConfig()
    encoder = GameStateEncoder()
    model = EuchreMLModel(config)

    # Load training data
    data_path = config.get_training_data_path("training_data.json")
    if not data_path.exists():
        print(f"Training data not found at {data_path}")
        print("Please run data collection scripts first.")
        return

    with open(data_path, "r") as f:
        examples = json.load(f)

    # Split data by action type and train separate models
    action_types = ["order_up", "call_trump", "play_card", "discard"]

    for action_type in action_types:
        dataset = EuchreDataset(examples, encoder, action_type)

        if len(dataset) == 0:
            print(f"No examples for {action_type}, skipping...")
            continue

        # Split into train/val
        train_size = int((1 - config.validation_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

        print(f"\nTraining {action_type} model...")
        print(f"  Training examples: {len(train_dataset)}")
        print(f"  Validation examples: {len(val_dataset)}")

        train_model(model, train_loader, val_loader, config, config.num_epochs, action_type)


if __name__ == "__main__":
    main()

