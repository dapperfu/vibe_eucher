"""Self-play training system for ML models."""

import uuid
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from eucher.game import Game
from eucher.training.data_collector import GameDataCollector
from eucher.training.profiling import timed_operation


class SelfPlayTrainer:
    """Trains ML models through self-play with various player combinations."""

    def __init__(
        self,
        data_collector: Optional[GameDataCollector] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        """
        Initialize the self-play trainer.

        Parameters
        ----------
        data_collector : Optional[GameDataCollector]
            Data collector instance. If None, creates a new one.
        output_dir : Optional[Path]
            Directory for output data. If None, uses default from MLConfig.
        """
        from eucher.players.computer.ml.ml_config import MLConfig

        self.config = MLConfig()
        if data_collector is None:
            if output_dir is None:
                output_dir = self.config.training_data_dir
            data_collector = GameDataCollector(output_dir)
        self.data_collector = data_collector

    @timed_operation("SelfPlayTrainer.run_game_with_collection")
    def run_game_with_collection(
        self, player_config: List[Tuple[str, str]], collect_from_players: Optional[List[int]] = None
    ) -> None:
        """
        Run a game and collect training data.

        Parameters
        ----------
        player_config : List[Tuple[str, str]]
            List of (name, profile_type) tuples for each player.
        collect_from_players : Optional[List[int]]
            List of player IDs to collect data from. If None, collects from all computer players.
        """
        game_id = str(uuid.uuid4())
        self.data_collector.start_game(game_id)

        # Create game
        game = Game(player_config)

        # Determine which players to collect from
        if collect_from_players is None:
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

            # Create wrapper methods that collect data
            def make_order_up_wrapper(pid: int):
                original = original_methods[pid]["decide_order_up"]

                def wrapper(p, tc, di, ts):
                    decision = original(p, tc, di, ts)
                    game_state = {
                        "trick_number": getattr(game, "_current_trick_number", 0),
                        "tricks_won_team0": getattr(game, "_current_tricks_won", [0, 0])[0],
                        "tricks_won_team1": getattr(game, "_current_tricks_won", [0, 0])[1],
                    }
                    self.data_collector.record_order_up_decision(
                        pid, p.hand, tc, di, decision, game_state
                    )
                    return decision

                return wrapper

            def make_call_trump_wrapper(pid: int):
                original = original_methods[pid]["decide_call_trump"]

                def wrapper(p, tc, ts, mc=False):
                    decision = original(p, tc, ts, mc)
                    game_state = {
                        "trick_number": getattr(game, "_current_trick_number", 0),
                        "tricks_won_team0": getattr(game, "_current_tricks_won", [0, 0])[0],
                        "tricks_won_team1": getattr(game, "_current_tricks_won", [0, 0])[1],
                    }
                    self.data_collector.record_call_trump_decision(
                        pid, p.hand, tc, decision, game_state
                    )
                    return decision

                return wrapper

            def make_play_card_wrapper(pid: int):
                original = original_methods[pid]["play_card"]

                def wrapper(p, ls, ts, tc, tpi):
                    decision = original(p, ls, ts, tc, tpi)
                    game_state = {
                        "trick_number": getattr(game, "_current_trick_number", 0),
                        "tricks_won_team0": getattr(game, "_current_tricks_won", [0, 0])[0],
                        "tricks_won_team1": getattr(game, "_current_tricks_won", [0, 0])[1],
                    }
                    self.data_collector.record_play_card_decision(
                        pid, p.hand, ls, ts, tc, decision, game_state
                    )
                    return decision

                return wrapper

            def make_discard_wrapper(pid: int):
                original = original_methods[pid]["choose_card_to_discard"]

                def wrapper(player, turned_card=None, ordered_up_by=None):
                    decision = original(player, turned_card, ordered_up_by)
                    game_state = {
                        "trick_number": getattr(game, "_current_trick_number", 0),
                        "tricks_won_team0": getattr(game, "_current_tricks_won", [0, 0])[0],
                        "tricks_won_team1": getattr(game, "_current_tricks_won", [0, 0])[1],
                    }
                    self.data_collector.record_discard_decision(pid, player.hand, decision, game_state)
                    return decision

                return wrapper

            # Replace methods with wrappers
            player.profile.decide_order_up = make_order_up_wrapper(player_idx)
            player.profile.decide_call_trump = make_call_trump_wrapper(player_idx)
            player.profile.play_card = make_play_card_wrapper(player_idx)
            player.profile.choose_card_to_discard = make_discard_wrapper(player_idx)

        # Play game
        tricks_won_history: List[List[int]] = []
        while True:
            continue_game = game.play_hand()
            tricks_won = getattr(game, "_current_tricks_won", [0, 0])
            tricks_won_history.append(tricks_won.copy())

            scores = game.get_scores()
            winner = game.get_winner()

            if winner is not None or not continue_game:
                break

        # Record game outcome
        self.data_collector.record_game_outcome(
            game_id, game.get_scores(), tricks_won_history, winner
        )
        
        # Save full game replay
        try:
            # Set game UUID so it can be saved
            game.game_uuid = game_id
            from eucher.game_file import save_game
            game_replay_file = save_game(game, self.data_collector.output_dir)
            self.data_collector.saved_game_replays.append(game_id)
        except Exception as e:
            # Don't fail if game replay save fails
            import sys
            print(f"Warning: Failed to save game replay for {game_id}: {e}", file=sys.stderr)

    @timed_operation("SelfPlayTrainer.run_training_round")
    def run_training_round(
        self,
        num_games: int = 100,
        player_combinations: Optional[List[List[Tuple[str, str]]]] = None,
        save_data: bool = True,
    ) -> None:
        """
        Run a round of training games with various player combinations.

        Parameters
        ----------
        num_games : int
            Number of games to run per combination.
        player_combinations : Optional[List[List[Tuple[str, str]]]]
            List of player configurations to test. If None, uses default combinations.
        save_data : bool
            Whether to save collected data at the end. Set False for batched saves.
        """
        if player_combinations is None:
            player_combinations = self._get_default_combinations()

        print(f"Running {len(player_combinations)} player combinations...")
        for combo_idx, combo in enumerate(player_combinations):
            print(f"\nCombination {combo_idx + 1}/{len(player_combinations)}: {combo}")
            for game_num in range(num_games):
                if (game_num + 1) % 10 == 0:
                    print(f"  Game {game_num + 1}/{num_games}")
                self.run_game_with_collection(combo)

        if save_data:
            print("\nSaving collected data...")
            saved_files = self.data_collector.save_data(use_uuid_naming=True)
            print(f"Saved {len(saved_files)} files with UUID-based naming")

    def _get_default_combinations(self) -> List[List[Tuple[str, str]]]:
        """
        Get default player combinations for training.

        Returns
        -------
        List[List[Tuple[str, str]]]
            List of player configurations.
        """
        return [
            # All ML players
            [("ML1", "ml_sklearn"), ("ML2", "ml_sklearn"), ("ML3", "ml_sklearn"), ("ML4", "ml_sklearn")],
            # ML vs Random
            [("ML1", "ml_sklearn"), ("Random1", "random"), ("ML2", "ml_sklearn"), ("Random2", "random")],
            # ML with AI partner vs Random opponents
            [("ML1", "ml_sklearn"), ("AI1", "ai"), ("Random1", "random"), ("Random2", "random")],
            # ML with Random partner vs AI and Random opponents
            [("ML1", "ml_sklearn"), ("Random1", "random"), ("AI1", "ai"), ("Random2", "random")],
            # ML vs Heuristic
            [("ML1", "ml_sklearn"), ("Heuristic1", "heuristic"), ("ML2", "ml_sklearn"), ("Heuristic2", "heuristic")],
            # Mixed: ML, AI, Random, Heuristic
            [("ML1", "ml_sklearn"), ("AI1", "ai"), ("Random1", "random"), ("Heuristic1", "heuristic")],
        ]


if __name__ == "__main__":
    trainer = SelfPlayTrainer()
    trainer.run_training_round(num_games=50)

