"""Train RL agents for Euchre decision making."""

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from eucher.game import Game
from eucher.players.computer.ml.ml_config import MLConfig
from eucher.players.computer.ml.ml_features import GameStateEncoder
from eucher.players.computer.ml.models.ml_models_rl import ExperienceReplayBuffer, RLAgent
from eucher.training.convergence_tracker import ConvergenceTracker
from eucher.training.training_orchestrator import TrainingOrchestrator


def calculate_reward(
    tricks_won: int,
    hand_won: bool,
    game_won: bool,
    euchred: bool,
    trick_number: int,
) -> float:
    """
    Calculate reward for an action.

    Parameters
    ----------
    tricks_won : int
        Number of tricks won in current hand.
    hand_won : bool
        Whether the hand was won.
    game_won : bool
        Whether the game was won.
    euchred : bool
        Whether the team was euchred (lost hand they called trump).
    trick_number : int
        Current trick number (0-4).

    Returns
    -------
    float
        Reward value.
    """
    reward = 0.0

    # Trick wins: +1 per trick
    reward += tricks_won * 1.0

    # Hand wins: +5
    if hand_won:
        reward += 5.0

    # Game wins: +20
    if game_won:
        reward += 20.0

    # Euchred: -5
    if euchred:
        reward -= 5.0

    # Small reward for making progress (later tricks worth slightly more)
    reward += trick_number * 0.1

    return reward


def train_rl_agent(
    agent: RLAgent,
    replay_buffer: ExperienceReplayBuffer,
    batch_size: int = 32,
    gamma: float = 0.99,
    target_update_freq: int = 100,
) -> float:
    """
    Train the RL agent on a batch from replay buffer.

    Parameters
    ----------
    agent : RLAgent
        The RL agent to train.
    replay_buffer : ExperienceReplayBuffer
        The experience replay buffer.
    batch_size : int
        Batch size for training.
    gamma : float
        Discount factor for future rewards.
    target_update_freq : int
        Update target network every N steps.

    Returns
    -------
    float
        Training loss value.
    """
    if len(replay_buffer) < batch_size:
        return 0.0

    # Sample batch
    batch = replay_buffer.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    # Convert to tensors
    states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(agent.device)
    actions_tensor = torch.tensor(actions, dtype=torch.long).to(agent.device)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(agent.device)
    next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32).to(agent.device)
    dones_tensor = torch.tensor(dones, dtype=torch.bool).to(agent.device)

    # Current Q values
    q_values = agent.q_network(states_tensor)
    q_value = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

    # Next Q values from target network
    with torch.no_grad():
        next_q_values = agent.target_network(next_states_tensor)
        next_q_value = next_q_values.max(1)[0]
        target_q_value = rewards_tensor + (gamma * next_q_value * ~dones_tensor)

    # Compute loss
    criterion = nn.MSELoss()
    loss = criterion(q_value, target_q_value)

    # Optimize
    optimizer = optim.Adam(agent.q_network.parameters(), lr=0.001)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update target network periodically
    agent.step_count = getattr(agent, "step_count", 0) + 1
    if agent.step_count % target_update_freq == 0:
        agent.target_network.load_state_dict(agent.q_network.state_dict())

    return loss.item()


def train_rl_through_self_play(
    num_games: int = 1000,
    config: Optional[MLConfig] = None,
    output_dir: Optional[Path] = None,
    checkpoint_interval: int = 100,
    convergence_tracker: Optional[ConvergenceTracker] = None,
    orchestrator: Optional[TrainingOrchestrator] = None,
) -> RLAgent:
    """
    Train RL agent through self-play.

    Parameters
    ----------
    num_games : int
        Number of games to play for training (used if orchestrator not provided).
    config : Optional[MLConfig]
        ML configuration. If None, creates a new one.
    output_dir : Optional[Path]
        Directory to save trained model and checkpoints.
    checkpoint_interval : int
        Save checkpoint every N games (used if orchestrator not provided).
    convergence_tracker : Optional[ConvergenceTracker]
        Convergence tracker for monitoring win rates.
    orchestrator : Optional[TrainingOrchestrator]
        Training orchestrator for coordinated training.

    Returns
    -------
    RLAgent
        Trained RL agent.
    """
    if config is None:
        config = MLConfig()

    if output_dir is None:
        output_dir = config.models_dir

    device = config.device
    print(f"Training RL agent on device: {device}")

    # Create agent
    agent = RLAgent(config)
    agent.step_count = 0

    # Create replay buffer
    replay_buffer = ExperienceReplayBuffer(capacity=10000)

    # Create encoder
    encoder = GameStateEncoder()

    # Use orchestrator if provided, otherwise use simple loop
    use_orchestrator = orchestrator is not None

    # Setup progress display
    from eucher.training.training_progress import TrainingProgressDisplay

    progress_display = None
    live_display = None

    if use_orchestrator:
        orchestrator.start_training()
        progress_display = TrainingProgressDisplay(
            orchestrator,
            trump_selection_risk=config.trump_selection_risk,
            gameplay_risk=config.gameplay_risk,
        )
        live_display = progress_display.start_live_display()
        live_display.__enter__()
    else:
        print(f"Training RL agent through {num_games} games of self-play...")

    total_rewards = []
    training_losses = []
    game_num = 0

    # Training statistics to track
    training_stats = {
        "games_played": 0,
        "total_rewards": [],
        "avg_reward": 0.0,
        "training_losses": [],
        "avg_loss": 0.0,
        "epsilon_history": [],
        "final_epsilon": 1.0,
    }

    try:
        while True:
            if use_orchestrator:
                if not orchestrator.should_continue():
                    break
            else:
                if game_num >= num_games:
                    break

            # Create game with RL agent
            player_config = [
                ("RL1", "ml"),
                ("RL2", "ml"),
                ("RL3", "ml"),
                ("RL4", "ml"),
            ]

            # For now, use a simple approach: play games and collect experiences
            # In a full implementation, we'd integrate RL agent into game loop
            game = Game(player_config)

            game_reward = 0.0
            episode_states = []
            episode_actions = []
            episode_rewards = []

            # Play game and collect experiences
            # This is a simplified version - full implementation would
            # integrate with game loop to collect state-action-reward tuples
            while True:
                continue_game = game.play_hand()
                # Calculate reward based on game outcome
                tricks_won = getattr(game, "_current_tricks_won", [0, 0])
                scores = game.get_scores()
                winner = game.get_winner()

                # Simplified: collect reward at end of hand
                if not continue_game or winner is not None:
                    break

            # Record game result
            won = winner is not None and winner == 0  # Simplified - would need actual team tracking
            if use_orchestrator:
                orchestrator.record_game_result(won)

            # Decay epsilon
            agent.update_epsilon()

            # Periodically train on replay buffer
            if len(replay_buffer) > 100:
                loss = train_rl_agent(agent, replay_buffer)
                if loss > 0:
                    training_losses.append(loss)
                    training_stats["training_losses"].append(loss)

            game_num += 1
            training_stats["games_played"] = game_num
            training_stats["epsilon_history"].append(agent.epsilon)

            # Update progress display
            if progress_display:
                progress_display.update()
                if game_num % 10 == 0:  # Update every 10 games for smoother display
                    progress_display.update_live_display(live_display)

            if game_num % 100 == 0 and not use_orchestrator:
                avg_reward = np.mean(total_rewards[-100:]) if total_rewards else 0.0
                avg_loss = np.mean(training_losses[-100:]) if training_losses else 0.0
                print(
                    f"Game [{game_num}] - Epsilon: {agent.epsilon:.3f}, Avg Reward: {avg_reward:.2f}, Avg Loss: {avg_loss:.4f}"
                )

            # Save checkpoint
            if use_orchestrator:
                if orchestrator.should_checkpoint():
                    current_stats = {
                        **training_stats,
                        "final_epsilon": agent.epsilon,
                        "avg_reward": np.mean(total_rewards[-100:]) if total_rewards else 0.0,
                        "avg_loss": np.mean(training_losses[-100:]) if training_losses else 0.0,
                    }
                    orchestrator.save_checkpoint(lambda p: agent.save(p, current_stats))
                    if progress_display:
                        progress_display.console.print(
                            f"[yellow]Checkpoint saved at game {game_num}[/yellow]"
                        )
            else:
                if output_dir is not None and game_num % checkpoint_interval == 0:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    checkpoint_path = output_dir / f"rl_checkpoint_game_{game_num}.pth"
                    current_stats = {
                        **training_stats,
                        "final_epsilon": agent.epsilon,
                        "avg_reward": np.mean(total_rewards[-100:]) if total_rewards else 0.0,
                        "avg_loss": np.mean(training_losses[-100:]) if training_losses else 0.0,
                    }
                    agent.save(checkpoint_path, current_stats)

    finally:
        # Close live display
        if live_display:
            live_display.__exit__(None, None, None)
            if progress_display:
                progress_display.print_summary()

    if not use_orchestrator:
        print("RL training complete!")

    # Update final statistics
    training_stats["final_epsilon"] = agent.epsilon
    training_stats["avg_reward"] = np.mean(total_rewards) if total_rewards else 0.0
    training_stats["avg_loss"] = np.mean(training_losses) if training_losses else 0.0

    # Save final model
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        final_path = output_dir / "rl_model.pth"
        agent.save(final_path, training_stats)
        print(f"Saved final model to {final_path}")
        print(f"Training statistics:")
        print(f"  Games played: {training_stats['games_played']}")
        print(f"  Final epsilon: {training_stats['final_epsilon']:.4f}")
        print(f"  Average reward: {training_stats['avg_reward']:.2f}")
        print(f"  Average loss: {training_stats['avg_loss']:.4f}")

    return agent


if __name__ == "__main__":
    from pathlib import Path

    config = MLConfig()
    train_rl_through_self_play(
        num_games=500,
        config=config,
        output_dir=config.models_dir,
    )

