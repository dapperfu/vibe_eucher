"""Self-play trainer for PyTorch AI."""

import signal
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from eucher.game import Game
from ..pytorch_networks import HybridNetwork
from ..pytorch_player import PyTorchStrategicPlayer
from .batch_tuner import tune_batch_size
from .checkpoint_manager import CheckpointManager
from .data_manager import TrainingDataManager
from .self_play_dashboard import SelfPlayDashboard
from .trainer import CumulativeTrainer, EuchreDataset
from eucher.training.data_collector import GameDataCollector
from eucher.training.self_play import SelfPlayTrainer


class SelfPlayPyTorchTrainer:
    """Self-play trainer for PyTorch AI that plays games and trains incrementally.

    Parameters
    ----------
    model : HybridNetwork
        PyTorch model to train.
    checkpoint_dir : str
        Directory for checkpoints.
    data_dir : str
        Directory for training data (optional, used for saving).
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
        """Initialize self-play trainer.

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
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model
        self.device = device
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        self.data_manager = TrainingDataManager(data_dir)
        self.cumulative_trainer = CumulativeTrainer(
            model=model, checkpoint_dir=checkpoint_dir, data_dir=data_dir, device=device
        )

        # Training state
        self.current_cycle = 0
        self.total_games_played = 0
        self.training_history: List[Dict] = []
        self.optimal_batch_size: Optional[int] = None

        # Dashboard (will be initialized in run_training_loop)
        self.dashboard: Optional[SelfPlayDashboard] = None

        # Signal handling for graceful shutdown
        self.interrupted = False
        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            if self.interrupted:
                # Already handling interrupt, force exit
                import sys
                sys.exit(1)
            print(f"\n\nReceived signal {signum}, saving checkpoint and exiting...")
            self.interrupted = True
            try:
                self.save_checkpoint()
                print("Checkpoint saved successfully")
            except Exception as e:
                print(f"Error saving checkpoint: {e}")
            import sys
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def save_checkpoint(self) -> None:
        """Save current training state to checkpoint.

        Saves model, training metrics, and state.
        Note: Optimizer state is not saved as we train incrementally.
        """
        checkpoint_data = {
            "model_state_dict": self.model.state_dict(),
            "current_cycle": self.current_cycle,
            "total_games_played": self.total_games_played,
            "optimal_batch_size": self.optimal_batch_size,
            "training_history": self.training_history,
            "epoch": self.current_cycle,  # Use cycle as epoch for compatibility
            "metrics": self.training_history[-1] if self.training_history else {},
            "training_duration": 0.0,  # Could track this if needed
        }

        checkpoint_path = self.checkpoint_manager.checkpoint_dir / f"self_play_checkpoint_cycle_{self.current_cycle}.pt"
        
        # Create temporary path for atomic write
        temp_path = checkpoint_path.with_suffix('.tmp')
        
        try:
            # Move model to CPU before saving to avoid CUDA errors during serialization
            model_device = next(self.model.parameters()).device
            model_on_cuda = model_device.type == "cuda"
            
            if model_on_cuda:
                # Temporarily move model to CPU for saving
                self.model.cpu()
            
            # Save to temporary file first
            torch.save(checkpoint_data, temp_path)
            
            # Atomic move: rename temp file to final checkpoint
            temp_path.replace(checkpoint_path)
            
            print(f"Checkpoint saved to: {checkpoint_path}")
            
            # Move model back to original device
            if model_on_cuda:
                self.model.to(model_device)
        except Exception as e:
            # If checkpoint saving fails, clean up temp file and log error
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            print(f"Warning: Failed to save checkpoint: {e}")
            # Try to move model back to original device if we moved it
            if model_on_cuda:
                try:
                    current_device = next(self.model.parameters()).device
                    if current_device.type != "cuda":
                        self.model.to(model_device)
                except Exception:
                    pass

    def load_checkpoint(self) -> bool:
        """Load latest checkpoint if available.

        Returns
        -------
        bool
            True if checkpoint was loaded, False otherwise.
        """
        # Find latest checkpoint
        checkpoint_files = list(self.checkpoint_manager.checkpoint_dir.glob("self_play_checkpoint_cycle_*.pt"))
        if not checkpoint_files:
            return False

        # Sort by cycle number
        checkpoint_files.sort(key=lambda p: int(p.stem.split("_")[-1]))

        latest_checkpoint = checkpoint_files[-1]
        print(f"Loading checkpoint from: {latest_checkpoint}")

        try:
            # Load to CPU first to avoid CUDA errors, then move to device
            checkpoint_data = torch.load(latest_checkpoint, map_location="cpu", weights_only=False)
        except Exception as e:
            # Checkpoint is corrupted, delete it and start fresh
            print(f"Warning: Failed to load checkpoint (possibly corrupted): {e}")
            print(f"Deleting corrupted checkpoint and starting fresh...")
            try:
                latest_checkpoint.unlink()
                print(f"Deleted corrupted checkpoint: {latest_checkpoint}")
            except Exception as delete_error:
                print(f"Warning: Could not delete corrupted checkpoint: {delete_error}")
            return False

        # Validate checkpoint data
        required_keys = ["model_state_dict", "current_cycle"]
        missing_keys = [key for key in required_keys if key not in checkpoint_data]
        if missing_keys:
            print(f"Warning: Checkpoint missing required keys: {missing_keys}")
            print("Starting fresh...")
            try:
                latest_checkpoint.unlink()
            except Exception:
                pass
            return False

        try:
            # Load model state dict (on CPU)
            self.model.load_state_dict(checkpoint_data["model_state_dict"])
            # Move model to target device
            self.model.to(self.device)
            self.model.eval()
            
            # Load training state
            self.current_cycle = checkpoint_data.get("current_cycle", 0)
            self.total_games_played = checkpoint_data.get("total_games_played", 0)
            self.optimal_batch_size = checkpoint_data.get("optimal_batch_size")
            self.training_history = checkpoint_data.get("training_history", [])
            
            print(f"Resumed from cycle {self.current_cycle}, total games: {self.total_games_played}")
            return True
        except Exception as e:
            # Model loading failed, checkpoint might be incompatible
            print(f"Warning: Failed to load model from checkpoint: {e}")
            print("Starting fresh...")
            return False

        # Restore model to eval mode
        self.model.eval()

        print(f"Resumed from cycle {self.current_cycle}, total games: {self.total_games_played}")
        return True

    def run_self_play_cycle(
        self, num_games: int, epsilon: float, player_config: List[Tuple[str, str]]
    ) -> GameDataCollector:
        """Run a cycle of self-play games with data collection.

        Parameters
        ----------
        num_games : int
            Number of games to play.
        epsilon : float
            Exploration probability for epsilon-greedy.
        player_config : List[Tuple[str, str]]
            Player configuration (name, profile_type).

        Returns
        -------
        GameDataCollector
            Collector with game data.
        """
        collector = GameDataCollector()

        # Convert player config to use pytorch_strategic for PyTorch players
        game_config = []
        for name, profile_type in player_config:
            if profile_type in ("pytorch", "ml_pytorch"):
                game_config.append((name, "pytorch_strategic"))
            else:
                game_config.append((name, profile_type))

        # Run games using SelfPlayTrainer but inject our custom profiles
        for game_num in range(num_games):
            if self.interrupted:
                break

            # Update dashboard
            if self.dashboard and hasattr(self, '_live_display'):
                self.dashboard.update_game_progress(game_num + 1, num_games)
                # Update display every 5 games to reduce overhead
                if (game_num + 1) % 5 == 0:
                    self.dashboard.update_live_display(self._live_display)

            # Create game first
            game = Game(game_config)

            # Update PyTorch players with current model and epsilon
            for i, (name, profile_type) in enumerate(player_config):
                if profile_type in ("pytorch", "ml_pytorch"):
                    # Replace profile with our custom one
                    profile = PyTorchStrategicPlayer(
                        model_path=None,
                        device=str(self.device),
                        exploration_epsilon=epsilon,
                    )
                    profile.model = self.model
                    profile.model.eval()
                    game.players[i].profile = profile

            # Determine which players to collect from
            collect_from_players = [
                i for i, (_, profile_type) in enumerate(player_config) if profile_type != "human"
            ]

            # Store original methods and create wrappers for data collection
            original_methods = {}
            for player_idx in collect_from_players:
                player = game.players[player_idx]
                original_methods[player_idx] = {
                    "decide_order_up": player.profile.decide_order_up,
                    "decide_call_trump": player.profile.decide_call_trump,
                    "play_card": player.profile.play_card,
                    "choose_card_to_discard": player.profile.choose_card_to_discard,
                }

            # Create wrapper methods that collect data (capture game and collector in closure)
            def make_order_up_wrapper(pid: int, game_ref, collector_ref, original_methods_ref):
                original = original_methods_ref[pid]["decide_order_up"]
                def wrapper(p, tc, di, ts):
                    decision = original(p, tc, di, ts)
                    game_state = {
                        "trick_number": getattr(game_ref, "_current_trick_number", 0),
                        "tricks_won_team0": getattr(game_ref, "_current_tricks_won", [0, 0])[0],
                        "tricks_won_team1": getattr(game_ref, "_current_tricks_won", [0, 0])[1],
                    }
                    collector_ref.record_order_up_decision(pid, p.hand, tc, di, decision, game_state)
                    return decision
                return wrapper

            def make_call_trump_wrapper(pid: int, game_ref, collector_ref, original_methods_ref):
                original = original_methods_ref[pid]["decide_call_trump"]
                def wrapper(p, tc, ts, mc=False):
                    decision = original(p, tc, ts, mc)
                    game_state = {
                        "trick_number": getattr(game_ref, "_current_trick_number", 0),
                        "tricks_won_team0": getattr(game_ref, "_current_tricks_won", [0, 0])[0],
                        "tricks_won_team1": getattr(game_ref, "_current_tricks_won", [0, 0])[1],
                    }
                    collector_ref.record_call_trump_decision(pid, p.hand, tc, decision, game_state)
                    return decision
                return wrapper

            def make_play_card_wrapper(pid: int, game_ref, collector_ref, original_methods_ref):
                original = original_methods_ref[pid]["play_card"]
                def wrapper(p, ls, ts, tc, tpi):
                    decision = original(p, ls, ts, tc, tpi)
                    from eucher.rules import RulesEngine
                    rules = RulesEngine()
                    is_renege = not rules.can_play_card(decision, p.hand, ls, ts)
                    game_state = {
                        "trick_number": getattr(game_ref, "_current_trick_number", 0),
                        "tricks_won_team0": getattr(game_ref, "_current_tricks_won", [0, 0])[0],
                        "tricks_won_team1": getattr(game_ref, "_current_tricks_won", [0, 0])[1],
                        "is_renege": is_renege,
                    }
                    collector_ref.record_play_card_decision(pid, p.hand, ls, ts, tc, decision, game_state)
                    return decision
                return wrapper

            def make_discard_wrapper(pid: int, game_ref, collector_ref, original_methods_ref):
                original = original_methods_ref[pid]["choose_card_to_discard"]
                def wrapper(player, turned_card=None, ordered_up_by=None):
                    decision = original(player, turned_card, ordered_up_by)
                    game_state = {
                        "trick_number": getattr(game_ref, "_current_trick_number", 0),
                        "tricks_won_team0": getattr(game_ref, "_current_tricks_won", [0, 0])[0],
                        "tricks_won_team1": getattr(game_ref, "_current_tricks_won", [0, 0])[1],
                    }
                    collector_ref.record_discard_decision(pid, player.hand, decision, game_state)
                    return decision
                return wrapper

            # Replace methods with wrappers
            for player_idx in collect_from_players:
                player = game.players[player_idx]
                player.profile.decide_order_up = make_order_up_wrapper(player_idx, game, collector, original_methods)
                player.profile.decide_call_trump = make_call_trump_wrapper(player_idx, game, collector, original_methods)
                player.profile.play_card = make_play_card_wrapper(player_idx, game, collector, original_methods)
                player.profile.choose_card_to_discard = make_discard_wrapper(player_idx, game, collector, original_methods)

            # Start game tracking
            game_id = str(game_num)
            collector.start_game(game_id)

            # Play game
            tricks_won_history = []
            while True:
                continue_game = game.play_hand()
                tricks_won = getattr(game, "_current_tricks_won", [0, 0])
                tricks_won_history.append(tricks_won.copy())

                scores = game.get_scores()
                winner = game.get_winner()

                if winner is not None or not continue_game:
                    break

            # Record game outcome
            renege_occurred = getattr(game, "_renege_occurred", False)
            renege_player_id = getattr(game, "_renege_player_id", None)
            collector.record_game_outcome(
                game_id, game.get_scores(), tricks_won_history, winner, renege_occurred, renege_player_id
            )

            # Update dashboard with game outcome
            if self.dashboard:
                scores = game.get_scores()
                self.dashboard.record_game_outcome(winner, scores, tricks_won_history)

        self.total_games_played += num_games

        # Update dashboard with sample counts
        if self.dashboard:
            self.dashboard.update_samples_collected(
                order_up=len(collector.order_up_data),
                trump_selection=len(collector.call_trump_data),
                play_card=len(collector.play_card_data),
                discard=len(collector.discard_data),
            )

        return collector

    def train_on_collected_data(
        self, collector: GameDataCollector, batch_size: Optional[int] = None, tune_batch: bool = False, learning_rate: float = 5e-5
    ) -> Dict:
        """Train model on collected data.

        Parameters
        ----------
        collector : GameDataCollector
            Data collector with game data.
        batch_size : Optional[int]
            Batch size to use. If None, uses optimal or default.
        tune_batch : bool
            If True, tune batch size automatically.
        learning_rate : float
            Learning rate for training (default: 5e-5).

        Returns
        -------
        Dict
            Training metrics.
        """
        # Convert collector data to training dataset
        training_data = self.data_manager.load_from_collector(collector)
        if not training_data:
            print("Warning: No training data collected")
            return {}

        # Create dataset
        dataset = EuchreDataset(training_data)

        if len(dataset) == 0:
            print("Warning: Dataset is empty")
            return {}

        # Determine batch size
        if tune_batch or self.optimal_batch_size is None:
            print("Tuning batch size...")
            self.optimal_batch_size = tune_batch_size(self.model, dataset, self.device)
            batch_size = self.optimal_batch_size
        elif batch_size is None:
            batch_size = self.optimal_batch_size or 32

        # Update dashboard training status
        if self.dashboard:
            self.dashboard.update_training_status(True)

        # Train for one epoch
        self.model.train()
        results = self.cumulative_trainer.train(
            dataset=dataset,
            num_epochs=1,
            learning_rate=learning_rate,
            batch_size=batch_size,
            resume=False,
            checkpoint_interval=999,  # Don't save during incremental training
        )

        self.model.eval()  # Set back to eval mode for inference

        # Update dashboard with training results
        final_metrics = results.get("final_metrics", {})
        if self.dashboard:
            loss = final_metrics.get("loss")
            if loss is not None:
                self.dashboard.record_loss(loss, learning_rate)
            self.dashboard.set_batch_size(batch_size)
            self.dashboard.update_training_status(False)

        return final_metrics

    def update_players_with_new_model(self, players: List[PyTorchStrategicPlayer]) -> None:
        """Update all PyTorch players with the current model.

        Parameters
        ----------
        players : List[PyTorchStrategicPlayer]
            List of players to update.
        """
        for player in players:
            player.model = self.model
            player.model.eval()

    def run_training_loop(
        self,
        games_per_training: int = 50,
        epsilon: float = 0.1,
        player_config: List[Tuple[str, str]] = None,
        max_cycles: Optional[int] = None,
        checkpoint_interval: int = 5,
        tune_batch: bool = False,
        resume: bool = True,
        learning_rate: float = 5e-5,
    ) -> None:
        """Run continuous self-play training loop.

        Parameters
        ----------
        games_per_training : int
            Number of games to play before each training step.
        epsilon : float
            Exploration probability.
        player_config : List[Tuple[str, str]]
            Player configuration. If None, uses all PyTorch players.
        max_cycles : Optional[int]
            Maximum training cycles. If None, runs indefinitely.
        checkpoint_interval : int
            Save checkpoint every N cycles.
        tune_batch : bool
            If True, automatically tune batch size.
        resume : bool
            If True, resume from checkpoint if available.
        learning_rate : float
            Learning rate for training (default: 5e-5).
        """
        # Load checkpoint if resuming
        if resume:
            self.load_checkpoint()

        # Default player config: all PyTorch
        if player_config is None:
            player_config = [
                ("PyTorch1", "pytorch"),
                ("PyTorch2", "pytorch"),
                ("PyTorch3", "pytorch"),
                ("PyTorch4", "pytorch"),
            ]

        # Initialize dashboard
        self.dashboard = SelfPlayDashboard(
            games_per_training=games_per_training,
            player_config=player_config,
        )

        # Start live display
        live_display = self.dashboard.start_live_display()
        self._live_display = live_display  # Store for access in run_self_play_cycle

        cycle = self.current_cycle
        try:
            with live_display:
                while True:
                    if self.interrupted:
                        break

                    if max_cycles is not None and cycle >= max_cycles:
                        break

                    # Update dashboard cycle info
                    self.dashboard.update_cycle(cycle + 1, self.total_games_played)
                    self.dashboard.update_live_display(live_display)

                    # Run self-play games
                    collector = self.run_self_play_cycle(games_per_training, epsilon, player_config)

                    # Update dashboard during game play
                    self.dashboard.update_live_display(live_display)

                    # Train on collected data
                    total_samples = len(collector.play_card_data) + len(collector.order_up_data) + len(collector.call_trump_data) + len(collector.discard_data)
                    metrics = self.train_on_collected_data(collector, tune_batch=tune_batch, learning_rate=learning_rate)

                    # Record metrics
                    cycle_metrics = {
                        "cycle": cycle + 1,
                        "games_played": games_per_training,
                        "total_games": self.total_games_played,
                        **metrics,
                    }
                    self.training_history.append(cycle_metrics)

                    # Update dashboard after training
                    self.dashboard.update_live_display(live_display)

                    # Save checkpoint periodically
                    if (cycle + 1) % checkpoint_interval == 0:
                        self.save_checkpoint()

                    cycle += 1
                    self.current_cycle = cycle
        finally:
            # Save final checkpoint
            self.save_checkpoint()

            # Print final summary
            if self.dashboard:
                self.dashboard.print_summary()

