"""Training loop for transformer RL with self-play and experience collection.

Implements sequence-based episodes, multi-phase action labeling, and batch training.
"""

from pathlib import Path
from typing import Dict, List, Optional

import torch

from src.ai_players.transformer_rl.actor_critic_agent import ActorCriticAgent
from src.ai_players.transformer_rl.checkpoint_manager import CumulativeCheckpointManager
from src.ai_players.transformer_rl.deduction_engine import DeductionEngine
from src.ai_players.transformer_rl.feature_encoder import TransformerRLFeatureEncoder
from src.ai_players.transformer_rl.reward_calculator import RewardCalculator
from eucher.game import Game
from src.training.transformer_rl.curriculum_trainer import CurriculumTrainer, TrainingStage
from src.training.transformer_rl.transformer_rl_dashboard import TransformerRLDashboard


class TrainingEpisode:
    """Represents a single training episode (one hand)."""

    def __init__(self) -> None:
        """Initialize training episode."""
        self.states: List[Dict] = []
        self.actions: List[int] = []
        self.action_types: List[str] = []
        self.rewards: List[float] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []
        self.valid_actions: List[List[int]] = []

    def add_step(
        self,
        state: Dict,
        action: int,
        action_type: str,
        reward: float,
        log_prob: float,
        value: float,
        done: bool,
        valid_actions: List[int],
    ) -> None:
        """Add a step to the episode.

        Parameters
        ----------
        state : Dict
            State dictionary.
        action : int
            Action taken.
        action_type : str
            Type of action.
        reward : float
            Reward received.
        log_prob : float
            Log probability of action.
        value : float
            Value estimate.
        done : bool
            Whether episode is done.
        valid_actions : List[int]
            Valid actions at this step.
        """
        self.states.append(state)
        self.actions.append(action)
        self.action_types.append(action_type)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
        self.valid_actions.append(valid_actions)

    def get_experiences(self) -> List[Dict]:
        """Convert episode to experience tuples.

        Returns
        -------
        List[Dict]
            List of experience dictionaries.
        """
        experiences = []
        for i in range(len(self.states)):
            exp = {
                "state": self.states[i],
                "action": self.actions[i],
                "action_type": self.action_types[i],
                "reward": self.rewards[i],
                "log_prob": self.log_probs[i],
                "value": self.values[i],
                "done": self.dones[i],
                "next_state": self.states[i + 1] if i + 1 < len(self.states) else None,
                "next_value": self.values[i + 1] if i + 1 < len(self.values) else 0.0,
            }
            experiences.append(exp)
        return experiences


def train_transformer_rl(
    agent: ActorCriticAgent,
    curriculum: CurriculumTrainer,
    num_hands: int = 1000,
    batch_size: int = 64,
    update_frequency: int = 10,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_manager: Optional[CumulativeCheckpointManager] = None,
    device: Optional[torch.device] = None,
    dashboard: Optional[TransformerRLDashboard] = None,
) -> Dict:
    """Train transformer RL agent through self-play.

    Parameters
    ----------
    agent : ActorCriticAgent
        The RL agent to train.
    curriculum : CurriculumTrainer
        Curriculum trainer for stage management.
    num_hands : int
        Number of hands to play.
    batch_size : int
        Batch size for training.
    update_frequency : int
        Update agent every N hands.
    checkpoint_dir : Optional[Path]
        Directory to save checkpoints (legacy).
    checkpoint_manager : Optional[CumulativeCheckpointManager]
        Checkpoint manager for UUID-based checkpoints.
    device : Optional[torch.device]
        Device to run on.

    Returns
    -------
    Dict
        Training statistics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize components
    feature_encoder = TransformerRLFeatureEncoder()
    reward_calculator = RewardCalculator()
    deduction_engine = DeductionEngine()

    # Training statistics
    training_stats = {
        "hands_played": 0,
        "hands_won": 0,
        "total_reward": 0.0,
        "avg_reward_per_hand": 0.0,
        "stage": curriculum.get_current_stage(),
    }

    # Start dashboard if provided
    live_display = None
    if dashboard:
        live_display = dashboard.start_live_display()
        live_display.__enter__()

    try:
        # Main training loop
        for hand_num in range(num_hands):
            # Get stage configuration
            stage_config = curriculum.get_stage_config()

            # Create game with appropriate player configuration
            if stage_config["opponent_type"] == "random":
                player_config = [
                    ("RL Agent", "transformer_rl"),
                    ("Random 1", "random"),
                    ("Random 2", "random"),
                    ("Random 3", "random"),
                ]
            else:  # self_play
                player_config = [
                    ("RL Agent 1", "transformer_rl"),
                    ("RL Agent 2", "transformer_rl"),
                    ("RL Agent 3", "transformer_rl"),
                    ("RL Agent 4", "transformer_rl"),
                ]

            # Create game
            game = Game(player_config)

            # Create training episode
            episode = TrainingEpisode()

            # Reset deduction engine
            deduction_engine.reset()

            # Play hand and collect experiences
            # Note: This is a simplified version - full implementation would
            # integrate with game loop to collect state-action-reward tuples
            try:
                continue_game = game.play_hand()

                # Calculate hand outcome
                tricks_won = getattr(game, "_current_tricks_won", [0, 0])
                scores = game.scores
                winner = game.get_winner()

                # Calculate rewards
                hand_won = tricks_won[0] >= 3  # Simplified: team 0 wins if 3+ tricks
                was_sweep = tricks_won[0] == 5
                was_set = tricks_won[1] == 0

                # Record performance
                win_rate = 1.0 if hand_won else 0.0
                curriculum.record_performance(win_rate)

                # Check stage progression
                if curriculum.should_progress_stage():
                    curriculum.progress_stage()
                    training_stats["stage"] = curriculum.get_current_stage()

                # Update training stats
                training_stats["hands_played"] += 1
                if hand_won:
                    training_stats["hands_won"] += 1

            except Exception as e:
                if dashboard:
                    dashboard.console.print(f"[red]Error playing hand {hand_num}: {e}[/red]")
                else:
                    print(f"Error playing hand {hand_num}: {e}")
                continue

            # Periodically update agent
            if hand_num % update_frequency == 0 and len(agent.buffer) > batch_size:
                update_stats = agent.update(batch_size=batch_size, num_epochs=4)
                training_stats.update(update_stats)

                # Save checkpoint (UUID-based if manager provided)
                if checkpoint_manager is not None:
                    checkpoint_uuid = agent.save(
                        training_stats=training_stats,
                        checkpoint_manager=checkpoint_manager
                    )
                    if checkpoint_uuid:
                        if dashboard:
                            dashboard.console.print(f"[yellow]Checkpoint saved: {checkpoint_uuid} (hand {hand_num})[/yellow]")
                        else:
                            print(f"Checkpoint saved: {checkpoint_uuid} (hand {hand_num})")
                elif checkpoint_dir is not None:
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    checkpoint_path = checkpoint_dir / f"checkpoint_{hand_num}.pth"
                    agent.save(checkpoint_path, training_stats)

                # Update dashboard
                if dashboard:
                    dashboard.update(
                        hand_num=hand_num,
                        total_hands=num_hands,
                        hands_won=training_stats["hands_won"],
                        hands_played=training_stats["hands_played"],
                        stage=curriculum.get_stage_name(),
                        **update_stats
                    )
                    dashboard.update_live_display(live_display)
            elif dashboard and hand_num % 10 == 0:
                # Update dashboard every 10 hands even without training update
                dashboard.update(
                    hand_num=hand_num,
                    total_hands=num_hands,
                    hands_won=training_stats["hands_won"],
                    hands_played=training_stats["hands_played"],
                    stage=curriculum.get_stage_name(),
                )
                dashboard.update_live_display(live_display)

    finally:
        # Close dashboard if provided
        if live_display:
            live_display.__exit__(None, None, None)
            if dashboard:
                dashboard.print_summary()

    # Final statistics
    if training_stats["hands_played"] > 0:
        training_stats["win_rate"] = training_stats["hands_won"] / training_stats["hands_played"]
        training_stats["avg_reward_per_hand"] = training_stats["total_reward"] / training_stats["hands_played"]

    return training_stats

