#!/usr/bin/env python3
"""Script to play a full game with 4 PyTorch AI players with different risk factors.

This script creates a game with 4 PyTorchStrategicPlayer instances, each with
different risk factors, and logs the results to a file.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eucher.ai_players.pytorch_player import PyTorchStrategicPlayer
from eucher.cards import Card, Suit
from eucher.game import Game
from eucher.players import Player
from eucher.players.profiles import PlayerProfile


class LoggingTUI:
    """TUI that logs gameplay details to GameLogger instead of printing."""

    def __init__(self, logger: "GameLogger") -> None:
        """Initialize logging TUI.

        Parameters
        ----------
        logger : GameLogger
            Logger to write gameplay details to.
        """
        self.logger = logger
        self.players: Optional[List[Player]] = None
        self.current_hand_log: List[str] = []
        self.dealer_id: Optional[int] = None
        self.trump_suit: Optional[Suit] = None

    def start_new_hand(self) -> None:
        """Start logging a new hand."""
        self.current_hand_log = []
        self.trump_suit = None

    def set_players(self, players: List[Player], dealer_id: Optional[int] = None) -> None:
        """Set the list of all players.

        Parameters
        ----------
        players : List[Player]
            List of all players in the game.
        dealer_id : Optional[int]
            The dealer's player ID, if available.
        """
        self.players = players
        self.dealer_id = dealer_id

    def log_initial_hands(self, players: List[Player]) -> None:
        """Log the initial hands dealt to all players.

        Parameters
        ----------
        players : List[Player]
            List of all players with their initial hands.
        """
        self.current_hand_log.append("")
        self.current_hand_log.append("INITIAL DEAL")
        self.current_hand_log.append("-" * 70)
        for player in players:
            cards = [str(card) for card in player.hand]
            # Format as 3 cards in first row, 2 cards in second row
            # First row: player name + first 3 cards
            first_row_cards = "  ".join(cards[:3])
            first_row = f"{player.name}: {first_row_cards}"
            self.current_hand_log.append(first_row)
            
            # Second row: indent to align with cards (account for player name + ": ")
            indent = len(player.name) + 2
            second_row_cards = "  ".join(cards[3:])
            second_row = " " * indent + second_row_cards
            self.current_hand_log.append(second_row)

    def log_turned_card(self, card: Card) -> None:
        """Log the turned up card.

        Parameters
        ----------
        card : Card
            The card that was turned up.
        """
        self.current_hand_log.append(f"Turned up card: {str(card)}")

    def log_trump_decision_history(self, order_up: list, call_trump: list) -> None:
        """Log all order-up and call-trump decisions for the hand.

        Parameters
        ----------
        order_up : list
            List of tuples `(player_name, bool)` representing order-up decisions.
        call_trump : list
            List of tuples `(player_name, Optional[Suit])` representing call-trump decisions.
        """
        if order_up:
            self.current_hand_log.append("")
            self.current_hand_log.append("Order Up Decisions:")
            for name, decision in order_up:
                action = "Ordered up" if decision else "Passed"
                self.current_hand_log.append(f"  {name}: {action}")

        if call_trump:
            self.current_hand_log.append("")
            self.current_hand_log.append("Call Trump Decisions:")
            for name, decision in call_trump:
                if decision is None:
                    self.current_hand_log.append(f"  {name}: Passed")
                else:
                    self.current_hand_log.append(f"  {name}: Called {decision.value} as trump")

    def log_trump_selected(self, trump_suit: Suit, maker_name: str) -> None:
        """Log that trump was selected.

        Parameters
        ----------
        trump_suit : Suit
            The selected trump suit.
        maker_name : str
            Name of the player who made trump.
        """
        self.trump_suit = trump_suit
        self.current_hand_log.append(f"Trump: {trump_suit.value} (made by {maker_name})")

    def log_trick(
        self,
        trick_num: int,
        played_cards: List[Card],
        player_ids: List[int],
        winner_id: int,
    ) -> None:
        """Log a trick with cards played and winner.

        Parameters
        ----------
        trick_num : int
            The trick number (1-5).
        played_cards : List[Card]
            Cards played in the trick.
        player_ids : List[int]
            Player IDs corresponding to each card.
        winner_id : int
            ID of the winning player.
        """
        if self.players is None:
            return

        self.current_hand_log.append("")
        self.current_hand_log.append(f"Trick {trick_num}:")
        for card, pid in zip(played_cards, player_ids):
            player_name = self.players[pid].name
            if pid == winner_id:
                # Make winning card bold using ANSI escape codes
                card_str = f"\033[1m{str(card)}\033[0m"
            else:
                card_str = str(card)
            self.current_hand_log.append(f"  {player_name}: {card_str}")

    def log_hand_score(
        self, tricks_won: List[int], scores: Tuple[int, int]
    ) -> None:
        """Log the hand score.

        Parameters
        ----------
        tricks_won : List[int]
            Tricks won by each team [team0, team1].
        scores : Tuple[int, int]
            Current scores (team0, team1).
        """
        self.current_hand_log.append("")
        self.current_hand_log.append(f"Tricks won: Team 0: {tricks_won[0]}, Team 1: {tricks_won[1]}")
        self.current_hand_log.append(f"Scores: Team 0: {scores[0]}, Team 1: {scores[1]}")

    def display_hand_log(self) -> None:
        """Display the current hand log and add it to logger."""
        if self.current_hand_log:
            for line in self.current_hand_log:
                self.logger.log(line)
            self.current_hand_log = []

    def display_trump_decision_summary(
        self,
        order_up: list,
        call_trump: list,
        trump_suit: Optional[Suit],
        maker_name: Optional[str],
    ) -> None:
        """Display a summary of all trump decisions (no-op for logging TUI).

        Parameters
        ----------
        order_up : list
            List of tuples `(player_name, bool)` representing order-up decisions.
        call_trump : list
            List of tuples `(player_name, Optional[Suit])` representing call-trump decisions.
        trump_suit : Optional[Suit]
            The selected trump suit.
        maker_name : Optional[str]
            Name of the player who made trump.
        """
        # Already logged via log_trump_decision_history, so no-op
        pass

    def display_trick_winner(self, winner_name: str) -> None:
        """Display the trick winner (no-op for logging TUI).

        Parameters
        ----------
        winner_name : str
            Name of the winning player.
        """
        # Already logged via log_trick, so no-op
        pass

    def update_trick_state(
        self, played_cards: List[Card], player_ids: List[int]
    ) -> None:
        """Update the current trick state (no-op for logging TUI).

        Parameters
        ----------
        played_cards : List[Card]
            Cards played in the current trick.
        player_ids : List[int]
            Player IDs corresponding to each card.
        """
        # Not needed for logging
        pass

    def update_trick_number(self, trick_number: int) -> None:
        """Update the current trick number (no-op for logging TUI).

        Parameters
        ----------
        trick_number : int
            The current trick number (0-4).
        """
        # Not needed for logging
        pass

    def display_scores(self, team0_score: int, team1_score: int) -> None:
        """Display current scores (no-op for logging TUI).

        Parameters
        ----------
        team0_score : int
            Team 0's score.
        team1_score : int
            Team 1's score.
        """
        # Scores are logged via log_hand_score
        pass

    def display_turned_card(self, card: Card) -> None:
        """Display the turned up card (no-op for logging TUI).

        Parameters
        ----------
        card : Card
            The turned card.
        """
        # Already logged via log_turned_card
        pass


class PyTorchPlayerProfile(PlayerProfile):
    """Wrapper profile that uses PyTorchStrategicPlayer with risk factors."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        trump_selection_risk: float = 0.5,
        gameplay_risk: float = 0.5,
        player_name: str = "PyTorch AI",
    ) -> None:
        """Initialize PyTorch player profile.

        Parameters
        ----------
        model_path : Optional[str]
            Path to model file.
        device : Optional[str]
            Device string.
        trump_selection_risk : float
            Risk factor for trump selection (0.0-1.0).
        gameplay_risk : float
            Risk factor for gameplay (0.0-1.0).
        player_name : str
            Name of the player.
        """
        super().__init__()
        self.player_name = player_name
        self.pytorch_player = PyTorchStrategicPlayer(
            model_path=model_path,
            device=device,
            trump_selection_risk=trump_selection_risk,
            gameplay_risk=gameplay_risk,
        )

    def decide_order_up(
        self, player: Player, turned_card: Card, dealer_id: int, trump_suit: Optional[Suit]
    ) -> bool:
        """Decide whether to order up."""
        return self.pytorch_player.decide_order_up(player, turned_card, dealer_id, trump_suit)

    def decide_call_trump(
        self,
        player: Player,
        turned_card: Card,
        trump_suit: Optional[Suit],
        must_choose: bool = False,
    ) -> Optional[Suit]:
        """Decide which suit to call as trump."""
        return self.pytorch_player.decide_call_trump(player, turned_card, trump_suit, must_choose)

    def choose_card_to_discard(
        self, player: Player, turned_card: Optional[Card] = None, ordered_up_by: Optional[str] = None
    ) -> Card:
        """Choose a card to discard (dealer only, after ordering up).

        Parameters
        ----------
        player : Player
            The dealer player.
        turned_card : Optional[Card]
            The card that was ordered up, if available.
        ordered_up_by : Optional[str]
            Name of the player who ordered up, if available.

        Returns
        -------
        Card
            The card to discard.
        """
        return self.pytorch_player.choose_card_to_discard(player, turned_card, ordered_up_by)

    def play_card(
        self,
        player: Player,
        led_suit: Optional[Suit],
        trump_suit: Optional[Suit],
        trick_cards: List[Card],
        trick_player_ids: List[int],
    ) -> Card:
        """Play a card."""
        return self.pytorch_player.play_card(
            player, led_suit, trump_suit, trick_cards, trick_player_ids
        )


class GameLogger:
    """Logger for game results."""

    def __init__(self, log_file: Path) -> None:
        """Initialize logger.

        Parameters
        ----------
        log_file : Path
            Path to log file.
        """
        self.log_file = log_file
        self.log_lines: List[str] = []

    def log(self, message: str) -> None:
        """Log a message.

        Parameters
        ----------
        message : str
            Message to log.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        self.log_lines.append(log_line)
        print(log_line)

    def save(self) -> None:
        """Save log to file."""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, "w") as f:
            f.write("\n".join(self.log_lines))
            f.write("\n")
        print(f"\nLog saved to: {self.log_file}")


def create_custom_game(
    risk_factors: List[Tuple[float, float]],
    model_path: Optional[str] = None,
    device: Optional[str] = None,
) -> Game:
    """Create a game with 4 PyTorch AI players with different risk factors.

    Parameters
    ----------
    risk_factors : List[Tuple[float, float]]
        List of (trump_selection_risk, gameplay_risk) tuples for each player.
    model_path : Optional[str]
        Path to model file.
    device : Optional[str]
        Device string.

    Returns
    -------
    Game
        Created game instance.
    """
    if len(risk_factors) != 4:
        raise ValueError("Must provide exactly 4 risk factor tuples")

    # Create custom profiles
    profiles: List[PlayerProfile] = []
    for i, (trump_risk, gameplay_risk) in enumerate(risk_factors):
        profile = PyTorchPlayerProfile(
            model_path=model_path,
            device=device,
            trump_selection_risk=trump_risk,
            gameplay_risk=gameplay_risk,
            player_name=f"PyTorch AI {i+1}",
        )
        profiles.append(profile)

    # Create players manually
    players: List[Player] = []
    for i, profile in enumerate(profiles):
        player = Player(f"PyTorch AI {i+1}", i, profile)
        players.append(player)

    # Create game with custom players
    # We'll need to manually set up the game structure
    game = Game.__new__(Game)  # Create without calling __init__
    game.players = players
    game.dealer_id = 0
    game.scores = [0, 0]
    game.trump_suit = None
    game.turned_card = None
    from eucher.rules import RulesEngine
    from eucher.trump import TrumpSelector
    from eucher.players.computer.ai import AIDecisionMaker

    game.rules = RulesEngine()
    game.trump_selector = None
    game.ai_decision_maker = AIDecisionMaker()
    game.tui = None
    game._ml_model = None
    game._ml_encoder = None
    game._current_trick_number = 0
    game._current_tricks_won = [0, 0]
    game.trump_selection_risk = None
    game.gameplay_risk = None

    return game


def play_pytorch_ai_game(
    risk_factors: List[Tuple[float, float]],
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    log_file: Optional[Path] = None,
) -> None:
    """Play a game with 4 PyTorch AI players.

    Parameters
    ----------
    risk_factors : List[Tuple[float, float]]
        List of (trump_selection_risk, gameplay_risk) tuples for each player.
    model_path : Optional[str]
        Path to model file.
    device : Optional[str]
        Device string.
    log_file : Optional[Path]
        Path to log file. If None, uses default.
    """
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path("logs") / f"pytorch_ai_game_{timestamp}.log"

    logger = GameLogger(log_file)

    logger.log("=" * 80)
    logger.log("PyTorch AI Game - 4 Players with Different Risk Factors")
    logger.log("=" * 80)
    logger.log("")

    # Log player configurations
    logger.log("Player Configurations:")
    for i, (trump_risk, gameplay_risk) in enumerate(risk_factors):
        logger.log(
            f"  Player {i+1}: Trump Risk={trump_risk:.2f}, Gameplay Risk={gameplay_risk:.2f}"
        )
    logger.log("")

    if model_path:
        logger.log(f"Using model: {model_path}")
    else:
        logger.log("Using default/random model")
    logger.log("")

    # Create game using standard Game class with custom profiles
    # We'll use a workaround: create profiles and inject them
    try:
        # Create profiles
        profiles: List[PlayerProfile] = []
        for i, (trump_risk, gameplay_risk) in enumerate(risk_factors):
            profile = PyTorchPlayerProfile(
                model_path=model_path,
                device=device,
                trump_selection_risk=trump_risk,
                gameplay_risk=gameplay_risk,
                player_name=f"PyTorch AI {i+1}",
            )
            profiles.append(profile)

        # Create game with placeholder config, then replace profiles
        player_config = [
            ("PyTorch AI 1", "heuristic"),
            ("PyTorch AI 2", "heuristic"),
            ("PyTorch AI 3", "heuristic"),
            ("PyTorch AI 4", "heuristic"),
        ]
        game = Game(player_config)

        # Replace profiles
        for i, profile in enumerate(profiles):
            game.players[i].profile = profile

        # Set up logging TUI to capture gameplay details
        logging_tui = LoggingTUI(logger)
        game.tui = logging_tui
        logging_tui.set_players(game.players, game.dealer_id)

        logger.log("Game created successfully")
        logger.log("")

        # Play game
        hand_num = 1
        while True:
            logger.log("=" * 80)
            logger.log(f"Hand {hand_num}")
            logger.log("=" * 80)

            # Update TUI with current dealer
            if game.tui is not None:
                game.tui.dealer_id = game.dealer_id

            continue_game = game.play_hand()

            # Hand log is already written by TUI.display_hand_log() during play_hand()
            # Just log the summary
            logger.log(f"Hand {hand_num} complete")
            logger.log(f"Current scores: Team 0 = {game.scores[0]}, Team 1 = {game.scores[1]}")
            logger.log("")

            # Check for game over
            winner = game.get_winner()
            if winner is not None:
                logger.log("=" * 80)
                logger.log("GAME OVER")
                logger.log("=" * 80)
                logger.log(f"Team {winner} wins!")
                logger.log(f"Final scores: Team 0 = {game.scores[0]}, Team 1 = {game.scores[1]}")
                logger.log("")
                break

            if not continue_game:
                logger.log("Game ended (no continue)")
                break

            hand_num += 1

        logger.log("=" * 80)
        logger.log("Game Summary")
        logger.log("=" * 80)
        logger.log(f"Total hands played: {hand_num}")
        logger.log(f"Final scores: Team 0 = {game.scores[0]}, Team 1 = {game.scores[1]}")
        winner = game.get_winner()
        if winner is not None:
            logger.log(f"Winner: Team {winner}")
        logger.log("")

        # Log player statistics
        logger.log("Player Risk Factors:")
        for i, (trump_risk, gameplay_risk) in enumerate(risk_factors):
            team = i % 2
            logger.log(
                f"  Player {i+1} (Team {team}): "
                f"Trump={trump_risk:.2f}, Gameplay={gameplay_risk:.2f}"
            )

    except Exception as e:
        logger.log(f"ERROR: Game failed with exception: {e}")
        import traceback

        logger.log(traceback.format_exc())
        raise

    finally:
        logger.save()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Play a game with 4 PyTorch AI players with different risk factors"
    )
    parser.add_argument(
        "--risk-factors",
        type=str,
        default="0.2,0.2,0.5,0.5,0.7,0.7,0.9,0.9",
        help=(
            "Comma-separated list of 8 risk factors: "
            "trump1,gameplay1,trump2,gameplay2,trump3,gameplay3,trump4,gameplay4 "
            "(default: 0.2,0.2,0.5,0.5,0.7,0.7,0.9,0.9)"
        ),
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to PyTorch model file (default: use default/random)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: 'cpu' or 'cuda' (default: auto-detect)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file (default: logs/pytorch_ai_game_TIMESTAMP.log)",
    )

    args = parser.parse_args()

    # Parse risk factors
    try:
        risk_values = [float(x.strip()) for x in args.risk_factors.split(",")]
        if len(risk_values) != 8:
            raise ValueError("Must provide exactly 8 risk factor values")
        risk_factors = [
            (risk_values[0], risk_values[1]),
            (risk_values[2], risk_values[3]),
            (risk_values[4], risk_values[5]),
            (risk_values[6], risk_values[7]),
        ]
    except ValueError as e:
        print(f"Error parsing risk factors: {e}")
        print("Expected format: trump1,gameplay1,trump2,gameplay2,trump3,gameplay3,trump4,gameplay4")
        return

    # Validate risk factors
    for i, (trump, gameplay) in enumerate(risk_factors):
        if not 0.0 <= trump <= 1.0:
            print(f"Error: Player {i+1} trump risk must be between 0.0 and 1.0")
            return
        if not 0.0 <= gameplay <= 1.0:
            print(f"Error: Player {i+1} gameplay risk must be between 0.0 and 1.0")
            return

    log_file = Path(args.log_file) if args.log_file else None

    play_pytorch_ai_game(
        risk_factors=risk_factors,
        model_path=args.model_path,
        device=args.device,
        log_file=log_file,
    )


if __name__ == "__main__":
    main()

