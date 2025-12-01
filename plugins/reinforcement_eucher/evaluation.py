"""Evaluation tools for ReinforcementEucher."""

from typing import Dict, List, Optional

import torch

from eucher.game import Game


class ReinforcementEucherEvaluator:
    """Evaluator for ReinforcementEucher performance.

    Parameters
    ----------
    model : ReinforcementEucherModel
        Model to evaluate.
    """

    def __init__(self, model) -> None:
        """Initialize evaluator.

        Parameters
        ----------
        model : ReinforcementEucherModel
            Model to evaluate.
        """
        self.model = model

    def evaluate_win_rate(
        self,
        num_games: int = 100,
        opponent_type: str = "random",
    ) -> Dict[str, float]:
        """Evaluate win rate against opponents.

        Parameters
        ----------
        num_games : int
            Number of games to play.
        opponent_type : str
            Opponent type ("random", "heuristic", "ai").

        Returns
        -------
        Dict[str, float]
            Evaluation metrics.
        """
        wins = 0
        losses = 0
        tricks_won_total = 0
        tricks_lost_total = 0

        for _ in range(num_games):
            # Create game with ReinforcementEucher players vs opponents
            player_config = [
                ("RE_0", "reinforcement_eucher"),
                ("Opp_1", opponent_type),
                ("RE_2", "reinforcement_eucher"),
                ("Opp_3", opponent_type),
            ]
            game = Game(player_config)

            # Set game references
            for player in game.players:
                if hasattr(player.profile, "set_game"):
                    player.profile.set_game(game)

            # Play game
            while True:
                continue_game = game.play_hand()
                if not continue_game:
                    break

            # Check winner
            winner = game.get_winner()
            if winner == 0:  # Team 0 (ReinforcementEucher players)
                wins += 1
            else:
                losses += 1

            # Track tricks (simplified)
            tricks_won_total += game.scores[0]
            tricks_lost_total += game.scores[1]

        win_rate = wins / num_games if num_games > 0 else 0.0
        avg_tricks_won = tricks_won_total / num_games if num_games > 0 else 0.0

        return {
            "win_rate": win_rate,
            "wins": wins,
            "losses": losses,
            "avg_tricks_won": avg_tricks_won,
            "num_games": num_games,
        }

    def evaluate_trick_win_rate(
        self,
        num_tricks: int = 1000,
    ) -> Dict[str, float]:
        """Evaluate trick win rate.

        Parameters
        ----------
        num_tricks : int
            Number of tricks to evaluate.

        Returns
        -------
        Dict[str, float]
            Trick evaluation metrics.
        """
        from eucher.players.computer.reinforcement_eucher.training.trick_simulator import TrickSimulator

        simulator = TrickSimulator(self.model.config)
        tricks_won = 0
        tricks_lost = 0

        for _ in range(num_tricks):
            scenario = simulator.generate_scenario()
            legal_actions = simulator.get_legal_actions(scenario)
            state = simulator._encode_trick_state(scenario, legal_actions).to(self.model.config.torch_device)
            legal_mask = torch.tensor(
                [1.0 if i in legal_actions else 0.0 for i in range(18)],
                device=self.model.config.torch_device,
            )

            with torch.no_grad():
                action, _, _ = self.model.sample_action(state.unsqueeze(0), legal_mask.unsqueeze(0))
                action_id = action.item()

            # Decode and simulate
            from eucher.players.computer.reinforcement_eucher.action_space import ActionEncoder
            action_type, card_idx, _, _ = ActionEncoder.decode_action(action_id)
            if card_idx is not None and card_idx < len(scenario.player_hand):
                played_card = scenario.player_hand[card_idx]
                won_trick, _ = simulator.simulate_trick_outcome(scenario, played_card)
                if won_trick:
                    tricks_won += 1
                else:
                    tricks_lost += 1

        total_tricks = tricks_won + tricks_lost
        win_rate = tricks_won / total_tricks if total_tricks > 0 else 0.0

        return {
            "trick_win_rate": win_rate,
            "tricks_won": tricks_won,
            "tricks_lost": tricks_lost,
            "num_tricks": total_tricks,
        }

    def compare_with_baseline(
        self,
        baseline_type: str = "heuristic",
        num_games: int = 100,
    ) -> Dict[str, float]:
        """Compare performance with baseline bot.

        Parameters
        ----------
        baseline_type : str
            Baseline bot type.
        num_games : int
            Number of games to play.

        Returns
        -------
        Dict[str, float]
            Comparison metrics.
        """
        re_metrics = self.evaluate_win_rate(num_games, opponent_type=baseline_type)
        return {
            "reinforcement_eucher_win_rate": re_metrics["win_rate"],
            "baseline_type": baseline_type,
            "num_games": num_games,
        }

