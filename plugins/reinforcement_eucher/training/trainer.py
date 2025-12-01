"""Main training orchestrator for ReinforcementEucher."""

import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from ..config import ReinforcementEucherConfig
from ..networks.model import ReinforcementEucherModel
from .dashboard import ReinforcementEucherDashboard
from .experience_buffer import ExperienceBuffer
from .ppo_trainer import PPOTrainer
from .self_play import SelfPlayGenerator
from .trick_simulator import TrickSimulator


def _convert_tensors_to_cpu(obj: any) -> any:
    """Recursively convert tensors in nested structures to CPU.

    Parameters
    ----------
    obj : any
        Object that may contain tensors.

    Returns
    -------
    any
        Object with tensors moved to CPU.
    """
    if isinstance(obj, torch.Tensor):
        return obj.cpu().detach()
    elif isinstance(obj, dict):
        return {k: _convert_tensors_to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_convert_tensors_to_cpu(item) for item in obj)
    else:
        return obj


def _convert_tensors_to_device(obj: any, device: torch.device) -> any:
    """Recursively convert tensors in nested structures to device.

    Parameters
    ----------
    obj : any
        Object that may contain tensors.
    device : torch.device
        Target device.

    Returns
    -------
    any
        Object with tensors moved to device.
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: _convert_tensors_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_convert_tensors_to_device(item, device) for item in obj)
    else:
        return obj


class ReinforcementEucherTrainer:
    """Main trainer coordinating trick-only and full-hand training stages.

    Parameters
    ----------
    model : ReinforcementEucherModel
        Model to train.
    config : ReinforcementEucherConfig
        Configuration.
    """

    def __init__(
        self,
        model: ReinforcementEucherModel,
        config: ReinforcementEucherConfig,
    ) -> None:
        """Initialize trainer.

        Parameters
        ----------
        model : ReinforcementEucherModel
            Model to train.
        config : ReinforcementEucherConfig
            Configuration.
        """
        self.model = model
        self.config = config
        self.device = config.torch_device

        # Training components
        self.ppo_trainer = PPOTrainer(model, config)
        self.experience_buffer = ExperienceBuffer(max_size=100000)
        self.trick_simulator = TrickSimulator(config)
        self.self_play_generator = SelfPlayGenerator(model, config)

        # Training state
        self.current_stage = config.training_stage
        self.game_count = 0
        self.trick_count = 0
        self.training_history: List[Dict] = []

        # Checkpoint management
        self.checkpoint_dir = config.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Dashboard
        self.dashboard: Optional[ReinforcementEucherDashboard] = None
        self.live_display = None

    def train_trick_only(self, num_tricks: int = 1000, use_dashboard: bool = True) -> Dict:
        """Train on trick-only scenarios (Stage 1).

        Parameters
        ----------
        num_tricks : int
            Number of trick scenarios to generate.
        use_dashboard : bool
            Whether to use Rich dashboard.

        Returns
        -------
        Dict
            Training metrics.
        """
        self.model.train()
        self.current_stage = "trick_only"

        # Initialize dashboard
        if use_dashboard and self.dashboard is None:
            self.dashboard = ReinforcementEucherDashboard(training_stage="trick_only")
            self.live_display = self.dashboard.start_live_display()
            self.live_display.__enter__()

        if self.dashboard:
            self.dashboard.update_stage("trick_only", 0, num_tricks)
            self.dashboard.update_training_progress(0, num_tricks)

        print(f"Starting trick-only training: {num_tricks} scenarios")

        for trick_num in range(num_tricks):
            # Generate random trick scenario
            scenario = self.trick_simulator.generate_scenario()
            legal_actions = self.trick_simulator.get_legal_actions(scenario)

            # Encode state
            state_tensor = self.trick_simulator._encode_trick_state(scenario, legal_actions).to(self.device)
            legal_mask = torch.tensor(
                [1.0 if i in legal_actions else 0.0 for i in range(18)],
                device=self.device,
                dtype=torch.float32,
            )

            # Ensure at least one legal action
            if legal_mask.sum() == 0:
                # Fallback: allow all actions if mask is empty
                legal_mask.fill_(1.0)

            # Sample action
            action, log_prob, value = self.model.sample_action(state_tensor.unsqueeze(0), legal_mask.unsqueeze(0))
            action_id = action.item()
            
            # Ensure action is legal (fallback if model sampled illegal action)
            if action_id not in legal_actions:
                action_id = legal_actions[0] if legal_actions else 0

            # Decode action to card
            from eucher.players.computer.reinforcement_eucher.action_space import ActionEncoder
            action_type, card_idx, suit, go_alone = ActionEncoder.decode_action(action_id)
            # Simplified: get card from hand
            if card_idx is not None and card_idx < len(scenario.player_hand):
                played_card = scenario.player_hand[card_idx]
            else:
                played_card = scenario.player_hand[0] if scenario.player_hand else None

            if played_card:
                # Simulate trick outcome
                won_trick, reward = self.trick_simulator.simulate_trick_outcome(scenario, played_card)

                # Store experience
                self.experience_buffer.add(
                    state=state_tensor,
                    action=action_id,
                    reward=reward,
                    advantage=reward - value.item(),
                    legal_mask=legal_mask,
                    value=value.item(),
                    log_prob=log_prob.item(),
                )

                self.trick_count += 1

            # Update dashboard with trick outcome
            if self.dashboard:
                self.dashboard.record_trick_outcome(won_trick)
                self.dashboard.update_stage("trick_only", self.trick_count, num_tricks)
                self.dashboard.update_training_progress(self.trick_count, num_tricks)

            # Train periodically
            if len(self.experience_buffer) >= self.config.batch_size:
                batch = self.experience_buffer.sample(self.config.batch_size)
                if self.dashboard:
                    self.dashboard.update_training_status(True)
                metrics = self.ppo_trainer.train_step(batch)
                if self.dashboard:
                    self.dashboard.update_training_status(False)
                
                self.training_history.append({
                    "trick": self.trick_count,
                    "stage": "trick_only",
                    **metrics,
                })

                # Update dashboard with training metrics
                if self.dashboard:
                    self.dashboard.record_losses(
                        policy_loss=metrics.get("policy_loss", 0.0),
                        value_loss=metrics.get("value_loss", 0.0),
                        entropy=metrics.get("entropy", 0.0),
                    )
                    self.dashboard.update_live_display(self.live_display)

                if self.trick_count % 100 == 0:
                    print(f"Trick {self.trick_count}: {metrics}")

        if self.dashboard and self.live_display:
            self.live_display.__exit__(None, None, None)
            self.dashboard.print_summary()

        return {
            "tricks_trained": self.trick_count,
            "stage": "trick_only",
        }

    def train_full_hand(self, num_games: int = 100) -> Dict:
        """Train on full hand games (Stage 2).

        Parameters
        ----------
        num_games : int
            Number of games to play.

        Returns
        -------
        Dict
            Training metrics.
        """
        self.model.train()
        self.current_stage = "full_hand"

        print(f"Starting full-hand training: {num_games} games")

        # This will be implemented when player profile is ready
        # For now, placeholder
        for game_num in range(num_games):
            # Generate self-play game
            # Collect experiences
            # Train on experiences
            self.game_count += 1

            if self.game_count % 10 == 0:
                print(f"Game {self.game_count}")

        return {
            "games_trained": self.game_count,
            "stage": "full_hand",
        }

    def save_checkpoint(
        self,
        suffix: Optional[str] = None,
        additional_state: Optional[Dict] = None,
    ) -> Path:
        """Save training checkpoint.

        Parameters
        ----------
        suffix : Optional[str]
            Optional suffix for checkpoint filename.
        additional_state : Optional[Dict]
            Additional state to save.

        Returns
        -------
        Path
            Path to saved checkpoint.
        """
        timestamp = int(time.time())
        suffix_str = f"_{suffix}" if suffix else ""
        checkpoint_name = f"checkpoint_game_{self.game_count}_trick_{self.trick_count}{suffix_str}_{timestamp}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.ppo_trainer.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "game_count": self.game_count,
            "trick_count": self.trick_count,
            "current_stage": self.current_stage,
            "training_history": self.training_history[-100:],  # Last 100 entries
            "timestamp": timestamp,
        }

        if additional_state:
            checkpoint["additional_state"] = additional_state

        # Save checkpoint with fallback for PyTorch serialization issues
        try:
            # Try torch.save with _use_new_zipfile_serialization=False to avoid serialization bug
            try:
                torch.save(checkpoint, checkpoint_path, _use_new_zipfile_serialization=False)
            except TypeError:
                # Parameter doesn't exist in this PyTorch version, try standard save
                torch.save(checkpoint, checkpoint_path)
        except (ModuleNotFoundError, AttributeError) as e:
            # Fallback to pickle if torch.save fails due to serialization issues
            if "serialization" in str(e).lower() or "torch.utils.serialization" in str(e):
                # Move all tensors to CPU for pickle compatibility
                checkpoint_cpu = _convert_tensors_to_cpu(checkpoint)
                
                # Save using pickle
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(checkpoint_cpu, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                # Re-raise if it's a different error
                raise

        # Save metadata
        metadata_path = checkpoint_path.with_suffix(".json")
        metadata = {
            "game_count": self.game_count,
            "trick_count": self.trick_count,
            "current_stage": self.current_stage,
            "checkpoint_path": str(checkpoint_path),
            "timestamp": timestamp,
            "training_history_length": len(self.training_history),
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Update latest checkpoint symlink
        latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
        if latest_path.exists():
            latest_path.unlink()
        latest_path.symlink_to(checkpoint_path.name)

        return checkpoint_path

    def load_checkpoint(
        self,
        checkpoint_path: Optional[Path] = None,
    ) -> Dict:
        """Load training checkpoint.

        Parameters
        ----------
        checkpoint_path : Optional[Path]
            Path to checkpoint. If None, loads latest.

        Returns
        -------
        Dict
            Checkpoint information.
        """
        if checkpoint_path is None:
            checkpoint_path = self._find_latest_checkpoint()

        if checkpoint_path is None or not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Try torch.load first, fallback to pickle if it fails
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except (ModuleNotFoundError, AttributeError, RuntimeError) as e:
            # Fallback to pickle if torch.load fails
            if "serialization" in str(e).lower() or "torch.utils.serialization" in str(e) or isinstance(e, RuntimeError):
                with open(checkpoint_path, "rb") as f:
                    checkpoint = pickle.load(f)
                # Convert all tensors back to proper device recursively
                checkpoint = _convert_tensors_to_device(checkpoint, self.device)
            else:
                # Re-raise if it's a different error
                raise

        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.ppo_trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load training state
        self.game_count = checkpoint.get("game_count", 0)
        self.trick_count = checkpoint.get("trick_count", 0)
        self.current_stage = checkpoint.get("current_stage", self.config.training_stage)
        self.training_history = checkpoint.get("training_history", [])

        return checkpoint

    def _find_latest_checkpoint(self) -> Optional[Path]:
        """Find the latest checkpoint.

        Returns
        -------
        Optional[Path]
            Path to latest checkpoint, or None if not found.
        """
        # Try latest symlink
        latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
        if latest_path.exists() and latest_path.is_symlink():
            resolved = latest_path.resolve()
            if resolved.exists():
                return resolved

        # Find most recent checkpoint file
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        if checkpoint_files:
            return max(checkpoint_files, key=lambda p: p.stat().st_mtime)

        return None

    def train(
        self,
        num_tricks: Optional[int] = None,
        num_games: Optional[int] = None,
        auto_resume: bool = True,
        use_dashboard: bool = True,
    ) -> Dict:
        """Main training loop.

        Parameters
        ----------
        num_tricks : Optional[int]
            Number of trick scenarios for Stage 1.
        num_games : Optional[int]
            Number of games for Stage 2.
        auto_resume : bool
            Whether to auto-resume from checkpoint.

        Returns
        -------
        Dict
            Training results.
        """
        # Try to resume from checkpoint
        if auto_resume:
            try:
                checkpoint = self.load_checkpoint()
                print(f"Resumed from checkpoint: game={self.game_count}, trick={self.trick_count}, stage={self.current_stage}")
            except FileNotFoundError:
                print("No checkpoint found, starting fresh")

        # Determine training stage
        if self.current_stage == "trick_only" or (num_tricks and num_tricks > 0):
            if num_tricks:
                self.train_trick_only(num_tricks, use_dashboard=use_dashboard)

        if self.current_stage == "full_hand" or (num_games and num_games > 0):
            if num_games:
                self.train_full_hand(num_games)

        # Save final checkpoint
        self.save_checkpoint(suffix="final")

        # Close dashboard if open
        if self.dashboard and self.live_display:
            self.live_display.__exit__(None, None, None)
            self.dashboard.print_summary()

        return {
            "game_count": self.game_count,
            "trick_count": self.trick_count,
            "current_stage": self.current_stage,
            "training_history": self.training_history,
        }

