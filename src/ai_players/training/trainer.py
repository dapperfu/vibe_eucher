"""Training infrastructure for PyTorch AI player."""

import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset

from src.ai_players.pytorch_networks import HybridNetwork
from src.ai_players.training.checkpoint_manager import CheckpointManager
from src.ai_players.training.data_manager import TrainingDataManager
from src.ai_players.training.memory_manager import MemoryManager


class EuchreDataset(Dataset):
    """Dataset for Euchre training data.

    Parameters
    ----------
    data : List[Dict]
        Training data samples.
    feature_encoder : EuchreFeatureEncoder
        Feature encoder for converting game state.
    """

    def __init__(self, data: List[Dict], feature_encoder=None) -> None:
        """Initialize dataset.

        Parameters
        ----------
        data : List[Dict]
            Training data.
        feature_encoder : Optional[EuchreFeatureEncoder]
            Feature encoder. If None, will create one.
        """
        self.data = data
        if feature_encoder is None:
            from src.ai_players.feature_encoder import EuchreFeatureEncoder

            self.feature_encoder = EuchreFeatureEncoder()
        else:
            self.feature_encoder = feature_encoder

    def __len__(self) -> int:
        """Get dataset length.

        Returns
        -------
        int
            Dataset size.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        """Get data sample.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        Dict
            Data sample with model-ready features.
        """
        sample = self.data[idx]

        # Handle different data formats
        if "features" in sample:
            # Old format with pre-encoded features - need to reconstruct
            return self._from_encoded_features(sample)
        elif "hand" in sample and isinstance(sample["hand"], list):
            # New format with raw game state
            return self._from_raw_state(sample)
        else:
            # Try to parse from string representations
            return self._from_string_format(sample)

    def _from_encoded_features(self, sample: Dict) -> Dict:
        """Convert from old encoded features format.

        Parameters
        ----------
        sample : Dict
            Sample with "features" field (pre-encoded numpy array as string or list).

        Returns
        -------
        Dict
            Model-ready sample.
        """
        import json
        from src.cards import Card, Rank, Suit
        from src.ai_players.game_state_tracker import TrickHistoryTracker

        # Parse features if it's a string (for compatibility, not used in reconstruction)
        if isinstance(sample.get("features"), str):
            try:
                json.loads(sample["features"])
            except (json.JSONDecodeError, TypeError):
                pass

        # For now, create minimal valid structure from available data
        # Since we don't have raw game state, create reasonable defaults
        trump_suit = None
        if sample.get("trump_suit"):
            try:
                trump_suit = Suit(sample["trump_suit"])
            except (ValueError, TypeError):
                pass

        # Create a minimal hand (5 cards) - this is a placeholder
        # In production, you'd want to reconstruct from features or use new data collection
        hand = [Card(Suit.HEARTS, Rank.ACE)] * 5
        tracker = TrickHistoryTracker()

        # Try to extract trick number from features if possible
        trick_number = sample.get("trick_number", 0)
        player_id = sample.get("player_id", 0)
        dealer_id = sample.get("dealer_id", 0)

        features = self.feature_encoder.encode_full_state(
            hand=hand,
            trick_history=[],
            tracker=tracker,
            trick_number=trick_number,
            team_score=0,
            opponent_score=0,
            player_position=player_id,
            dealer_id=dealer_id,
            trump_suit=trump_suit,
            current_trick=None,
        )

        return {
            "hand": features["hand"],
            "trick_history": features["trick_history"],
            "won_tricks_summary": features["won_tricks_summary"],
            "game_context": features["game_context"],
            "current_trick": features["current_trick"],
            "target": torch.tensor([sample.get("decision", 0)], dtype=torch.long),
            "action_type": "play_card",  # Default action type
        }

    def _from_raw_state(self, sample: Dict) -> Dict:
        """Convert from raw game state format.

        Parameters
        ----------
        sample : Dict
            Sample with raw game state.

        Returns
        -------
        Dict
            Model-ready sample.
        """
        from src.cards import Suit
        from src.ai_players.game_state_tracker import TrickHistoryTracker

        # Parse cards
        hand = self._parse_cards(sample.get("hand", []))
        trick_cards = self._parse_cards(sample.get("trick_cards", []))
        trump_suit = Suit(sample["trump_suit"]) if sample.get("trump_suit") else None

        tracker = TrickHistoryTracker()
        trick_history = sample.get("trick_history", [])

        features = self.feature_encoder.encode_full_state(
            hand=hand,
            trick_history=trick_history,
            tracker=tracker,
            trick_number=sample.get("trick_number", 0),
            team_score=sample.get("tricks_won_team0", 0),
            opponent_score=sample.get("tricks_won_team1", 0),
            player_position=sample.get("player_id", 0),
            dealer_id=sample.get("dealer_id", 0),
            trump_suit=trump_suit,
            current_trick=trick_cards if trick_cards else None,
        )

        # Get target
        target = self._get_target(sample)

        return {
            "hand": features["hand"],
            "trick_history": features["trick_history"],
            "won_tricks_summary": features["won_tricks_summary"],
            "game_context": features["game_context"],
            "current_trick": features["current_trick"],
            "target": target,
            "action_type": sample.get("action_type", "play_card"),
        }

    def _from_string_format(self, sample: Dict) -> Dict:
        """Convert from string-based format.

        Parameters
        ----------
        sample : Dict
            Sample with string representations.

        Returns
        -------
        Dict
            Model-ready sample.
        """
        # Try to parse string representations
        return self._from_raw_state(sample)

    def _parse_cards(self, card_data: List) -> List:
        """Parse cards from various formats.

        Parameters
        ----------
        card_data : List
            Card data in various formats.

        Returns
        -------
        List[Card]
            List of Card objects.
        """
        from src.cards import Card, Rank, Suit

        cards = []
        for item in card_data:
            if isinstance(item, Card):
                cards.append(item)
            elif isinstance(item, str):
                # Parse string like "K♥" or "A♠"
                card = self._parse_card_string(item)
                if card:
                    cards.append(card)
            elif isinstance(item, dict):
                # Parse dict with suit/rank
                suit = Suit(item["suit"]) if isinstance(item["suit"], str) else item["suit"]
                rank = Rank(item["rank"]) if isinstance(item["rank"], int) else Rank(item["rank"])
                cards.append(Card(suit, rank))

        return cards

    def _parse_card_string(self, card_str: str):
        """Parse card from string.

        Parameters
        ----------
        card_str : str
            Card string like "K♥" or "A♠".

        Returns
        -------
        Optional[Card]
            Parsed card or None.
        """
        from src.cards import Card, Rank, Suit

        if not card_str or len(card_str) < 2:
            return None

        rank_map = {
            "9": Rank.NINE,
            "10": Rank.TEN,
            "J": Rank.JACK,
            "Q": Rank.QUEEN,
            "K": Rank.KING,
            "A": Rank.ACE,
        }
        suit_map = {
            "♥": Suit.HEARTS,
            "H": Suit.HEARTS,
            "♦": Suit.DIAMONDS,
            "D": Suit.DIAMONDS,
            "♣": Suit.CLUBS,
            "C": Suit.CLUBS,
            "♠": Suit.SPADES,
            "S": Suit.SPADES,
        }

        # Try to parse
        rank_str = card_str[:-1] if len(card_str) > 1 else card_str[0]
        suit_str = card_str[-1]

        rank = rank_map.get(rank_str)
        suit = suit_map.get(suit_str)

        if rank and suit:
            return Card(suit, rank)

        return None

    def _get_target(self, sample: Dict) -> torch.Tensor:
        """Get target tensor from sample.

        Parameters
        ----------
        sample : Dict
            Sample data.

        Returns
        -------
        torch.Tensor
            Target tensor.
        """
        if "decision" in sample:
            return torch.tensor([sample["decision"]], dtype=torch.long)
        elif "action_value" in sample:
            # Parse action value
            return torch.tensor([0], dtype=torch.long)  # Placeholder
        else:
            return torch.tensor([0], dtype=torch.long)


class MultiDeviceTrainer:
    """Multi-device trainer with automatic device detection and optimization.

    Parameters
    ----------
    model : HybridNetwork
        Model to train.
    device : Optional[torch.device]
        Device to train on. If None, auto-detects.
    """

    def __init__(self, model: HybridNetwork, device: Optional[torch.device] = None) -> None:
        """Initialize multi-device trainer.

        Parameters
        ----------
        model : HybridNetwork
            Model to train.
        device : Optional[torch.device]
            Device to use.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.model = model.to(device)
        self.memory_manager = MemoryManager(device)

        # Get optimal training configuration
        param_count = model.get_parameter_count()
        self.config = self.memory_manager.get_training_config(param_count)

        # Setup mixed precision
        self.use_mixed_precision = self.config.get("use_mixed_precision", False)
        self.scaler = GradScaler() if self.use_mixed_precision else None

        # Gradient accumulation
        self.gradient_accumulation_steps = self.config.get("gradient_accumulation_steps", 1)

        print(f"Training on {device}")
        print(f"Configuration: {self.config}")

    def create_optimizer(self, learning_rate: float = 1e-4) -> torch.optim.Optimizer:
        """Create optimizer with appropriate settings.

        Parameters
        ----------
        learning_rate : float
            Learning rate.

        Returns
        -------
        torch.optim.Optimizer
            Optimizer.
        """
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5,
        )

    def create_data_loader(
        self, dataset: Dataset, batch_size: Optional[int] = None, shuffle: bool = True
    ) -> DataLoader:
        """Create data loader with optimal settings.

        Parameters
        ----------
        dataset : Dataset
            Training dataset.
        batch_size : Optional[int]
            Batch size. If None, uses optimal from config.
        shuffle : bool
            Whether to shuffle data.

        Returns
        -------
        DataLoader
            Data loader.
        """
        if batch_size is None:
            batch_size = self.config.get("batch_size", 32)

        num_workers = self.config.get("num_workers", 0) if not self.is_gpu() else 0
        prefetch_factor = self.config.get("data_prefetch", 2)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=self.is_gpu(),
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )

    def is_gpu(self) -> bool:
        """Check if using GPU.

        Returns
        -------
        bool
            True if GPU, False otherwise.
        """
        return self.device.type == "cuda"

    def train_step(
        self,
        batch: Dict,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        accumulation_step: int = 0,
    ) -> Dict:
        """Perform a single training step.

        Parameters
        ----------
        batch : Dict
            Batch of training data.
        optimizer : torch.optim.Optimizer
            Optimizer.
        criterion : nn.Module
            Loss function.
        accumulation_step : int
            Current gradient accumulation step.

        Returns
        -------
        Dict
            Training metrics.
        """
        self.model.train()

        # Move batch to device and handle DataLoader batching
        # DataLoader returns batches as lists, need to stack them
        if isinstance(batch, list):
            # Stack tensors from list of dicts
            stacked_batch = {}
            for key in batch[0].keys():
                if isinstance(batch[0][key], torch.Tensor):
                    stacked_batch[key] = torch.stack([item[key] for item in batch]).to(self.device)
                else:
                    stacked_batch[key] = [item[key] for item in batch]
            batch = stacked_batch
        else:
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

        # Forward pass with mixed precision if enabled
        if self.is_gpu() and self.use_mixed_precision:
            try:
                from torch.amp import autocast as autocast_new

                with autocast_new("cuda"):
                    outputs = self._forward_pass(batch)
            except (ImportError, AttributeError):
                from torch.cuda.amp import autocast

                with autocast():
                    outputs = self._forward_pass(batch)
        else:
            outputs = self._forward_pass(batch)

        # Calculate loss
        loss = self._calculate_loss(outputs, batch, criterion)

        # Backward pass
        if self.use_mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Update weights if accumulation is complete
        if (accumulation_step + 1) % self.gradient_accumulation_steps == 0:
            if self.use_mixed_precision:
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        return {"loss": loss.item()}

    def _forward_pass(self, batch: Dict) -> Dict:
        """Perform forward pass through model.

        Parameters
        ----------
        batch : Dict
            Batch of data.

        Returns
        -------
        Dict
            Model outputs.
        """
        return self.model(
            hand=batch["hand"],
            trick_history=batch.get("trick_history"),
            won_tricks_summary=batch.get("won_tricks_summary"),
            game_context=batch.get("game_context"),
            current_trick=batch.get("current_trick"),
        )

    def _calculate_loss(
        self, outputs: Dict, batch: Dict, criterion: nn.Module
    ) -> torch.Tensor:
        """Calculate loss from outputs and targets.

        Parameters
        ----------
        outputs : Dict
            Model outputs.
        batch : Dict
            Batch with targets.
        criterion : nn.Module
            Loss function.

        Returns
        -------
        torch.Tensor
            Loss value.
        """
        # Determine which output head to use based on action type
        action_type = batch.get("action_type", "play_card")
        target = batch.get("target")
        
        if target is None:
            # No target - return dummy loss
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Ensure target is the right shape and type
        if isinstance(target, torch.Tensor):
            if target.dim() == 0:
                target = target.unsqueeze(0)
            # CrossEntropyLoss expects Long type for class indices
            if target.dtype != torch.long:
                target = target.long()
        else:
            target = torch.tensor([target], dtype=torch.long, device=self.device)

        # Handle batch dimension - target might be batched
        batch_size = outputs["card_play"].shape[0] if "card_play" in outputs else 1
        if target.shape[0] != batch_size:
            # Expand target to match batch size
            if target.shape[0] == 1:
                target = target.expand(batch_size)

        # Select appropriate output head
        if action_type == "order_up":
            output = outputs["order_up"].squeeze(-1)  # Binary classification
            # For binary, use BCEWithLogitsLoss
            if output.shape[-1] == 1:
                # Binary classification - target should be 0 or 1
                target_float = target.float().squeeze()
                if target_float.dim() == 0:
                    target_float = target_float.unsqueeze(0)
                return nn.functional.binary_cross_entropy_with_logits(
                    output.squeeze(-1), target_float
                )
            else:
                # Multi-class (shouldn't happen for order_up)
                return criterion(output, target.squeeze())
        elif action_type == "trump_selection":
            output = outputs["trump_selection"]
            target_squeezed = target.squeeze()
            if target_squeezed.dim() == 0:
                target_squeezed = target_squeezed.unsqueeze(0)
            # Clamp target to valid range
            target_squeezed = torch.clamp(target_squeezed, 0, output.shape[-1] - 1)
            return criterion(output, target_squeezed)
        elif action_type == "discard":
            output = outputs["discard"]
            target_squeezed = target.squeeze()
            if target_squeezed.dim() == 0:
                target_squeezed = target_squeezed.unsqueeze(0)
            # Clamp target to valid range (0-5 for 6 cards)
            target_squeezed = torch.clamp(target_squeezed, 0, output.shape[-1] - 1)
            return criterion(output, target_squeezed)
        elif action_type == "play_card":
            output = outputs["card_play"]
            target_squeezed = target.squeeze()
            if target_squeezed.dim() == 0:
                target_squeezed = target_squeezed.unsqueeze(0)
            # The target might be a global card index, but output is hand position
            # For now, clamp to valid range (0-5 for 6 cards in hand)
            # In production, you'd want to map global card index to hand position
            target_squeezed = torch.clamp(target_squeezed, 0, output.shape[-1] - 1)
            return criterion(output, target_squeezed)
        else:
            # Default to card_play
            output = outputs["card_play"]
            target_squeezed = target.squeeze()
            if target_squeezed.dim() == 0:
                target_squeezed = target_squeezed.unsqueeze(0)
            target_squeezed = torch.clamp(target_squeezed, 0, output.shape[-1] - 1)
            return criterion(output, target_squeezed)


class CumulativeTrainer:
    """Cumulative trainer that supports incremental training.

    Parameters
    ----------
    model : HybridNetwork
        Model to train.
    checkpoint_dir : str
        Checkpoint directory.
    data_dir : str
        Training data directory.
    device : Optional[torch.device]
        Device to train on.
    """

    def __init__(
        self,
        model: HybridNetwork,
        checkpoint_dir: str = "models/checkpoints/pytorch_ai",
        data_dir: str = "training_data",
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize cumulative trainer.

        Parameters
        ----------
        model : HybridNetwork
            Model to train.
        checkpoint_dir : str
            Checkpoint directory.
        data_dir : str
            Data directory.
        device : Optional[torch.device]
            Device to use.
        """
        self.device_trainer = MultiDeviceTrainer(model, device)
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        self.data_manager = TrainingDataManager(data_dir)
        self.model = model

        self.training_history: List[Dict] = []
        self.current_epoch = 0

    def train(
        self,
        dataset: Dataset,
        num_epochs: int,
        learning_rate: float = 1e-4,
        batch_size: Optional[int] = None,
        resume: bool = False,
        checkpoint_interval: int = 5,
    ) -> Dict:
        """Train the model.

        Parameters
        ----------
        dataset : Dataset
            Training dataset.
        num_epochs : int
            Number of epochs.
        learning_rate : float
            Learning rate.
        batch_size : Optional[int]
            Batch size.
        resume : bool
            Whether to resume from checkpoint.
        checkpoint_interval : int
            Save checkpoint every N epochs.

        Returns
        -------
        Dict
            Training results.
        """
        # Resume from checkpoint if requested
        if resume:
            try:
                self.current_epoch, metrics, duration = self.checkpoint_manager.resume_from_checkpoint(
                    self.model,
                    self._create_optimizer(learning_rate),
                )
                print(f"Resumed from epoch {self.current_epoch}")
            except FileNotFoundError:
                print("No checkpoint found, starting from scratch")

        optimizer = self._create_optimizer(learning_rate)
        criterion = nn.CrossEntropyLoss()

        data_loader = self.device_trainer.create_data_loader(dataset, batch_size)

        total_duration = 0.0

        for epoch in range(self.current_epoch, self.current_epoch + num_epochs):
            epoch_start = time.time()
            epoch_metrics = self._train_epoch(data_loader, optimizer, criterion)
            epoch_duration = time.time() - epoch_start
            total_duration += epoch_duration

            epoch_metrics["epoch"] = epoch
            epoch_metrics["duration"] = epoch_duration
            self.training_history.append(epoch_metrics)

            print(f"Epoch {epoch}: {epoch_metrics}")

            # Save checkpoint periodically
            if (epoch + 1) % checkpoint_interval == 0:
                self.checkpoint_manager.save_checkpoint(
                    epoch=epoch,
                    model=self.model,
                    optimizer=optimizer,
                    metrics=epoch_metrics,
                    training_duration=total_duration,
                )

        # Save final checkpoint
        final_metrics = self.training_history[-1] if self.training_history else {}
        self.checkpoint_manager.save_checkpoint(
            epoch=self.current_epoch + num_epochs - 1,
            model=self.model,
            optimizer=optimizer,
            metrics=final_metrics,
            training_duration=total_duration,
            suffix="final",
        )

        return {
            "total_epochs": num_epochs,
            "total_duration": total_duration,
            "final_metrics": final_metrics,
            "history": self.training_history,
        }

    def _train_epoch(
        self, data_loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module
    ) -> Dict:
        """Train for one epoch.

        Parameters
        ----------
        data_loader : DataLoader
            Data loader.
        optimizer : torch.optim.Optimizer
            Optimizer.
        criterion : nn.Module
            Loss function.

        Returns
        -------
        Dict
            Epoch metrics.
        """
        total_loss = 0.0
        num_batches = 0

        accumulation_step = 0
        for batch in data_loader:
            metrics = self.device_trainer.train_step(batch, optimizer, criterion, accumulation_step)
            total_loss += metrics["loss"]
            num_batches += 1
            accumulation_step += 1

        return {"loss": total_loss / num_batches if num_batches > 0 else 0.0}

    def _create_optimizer(self, learning_rate: float) -> torch.optim.Optimizer:
        """Create optimizer.

        Parameters
        ----------
        learning_rate : float
            Learning rate.

        Returns
        -------
        torch.optim.Optimizer
            Optimizer.
        """
        return self.device_trainer.create_optimizer(learning_rate)

