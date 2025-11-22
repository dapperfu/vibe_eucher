"""Main game logic and state management for Euchre."""

from typing import List, Optional, Tuple

from src.ai import AIDecisionMaker
from src.cards import Card, Deck, Suit
from src.ml_config import MLConfig
from src.ml_features import GameStateEncoder
from src.ml_model import EuchreMLModel
from src.ml_player import MLPlayer
from src.player_profiles import (
    AIPlayer,
    HeuristicPlayer,
    HumanProfile,
    MLBasedProfile,
    PlayerProfile,
    RandomPlayer,
)
from src.players import Player
from src.rules import RulesEngine
from src.trump import TrumpSelector


class Game:
    """Manages the Euchre game state and flow."""

    def __init__(
        self,
        player_config: List[Tuple[str, str]],
        trump_selection_risk: Optional[float] = None,
        gameplay_risk: Optional[float] = None,
    ) -> None:
        """
        Initialize a new game.

        Parameters
        ----------
        player_config : List[Tuple[str, str]]
            List of (name, profile_type) tuples for each player.
            profile_type can be: "human", "simple", "ai", "random", "ml", "ml_sklearn", "ml_pytorch"
        trump_selection_risk : Optional[float]
            Risk factor for trump selection decisions (0.0-1.0) for ML players.
            If None, uses MLConfig default.
        gameplay_risk : Optional[float]
            Risk factor for gameplay decisions (0.0-1.0) for ML players.
            If None, uses MLConfig default.
        """
        if len(player_config) != 4:
            raise ValueError("Euchre requires exactly 4 players")

        self.players: List[Player] = []
        self.dealer_id: int = 0
        self.scores: List[int] = [0, 0]  # Team 0 and Team 1 scores
        self.trump_suit: Optional[Suit] = None
        self.turned_card: Optional[Card] = None
        self.rules = RulesEngine()
        self.trump_selector: Optional[TrumpSelector] = None
        self.ai_decision_maker = AIDecisionMaker()
        self.tui = None

        # ML model setup (lazy initialization)
        self._ml_model: Optional[EuchreMLModel] = None
        self._ml_encoder: Optional[GameStateEncoder] = None

        # Game state tracking for ML profiles
        self._current_trick_number: int = 0
        self._current_tricks_won: List[int] = [0, 0]

        # Store risk factors for ML players
        self.trump_selection_risk = trump_selection_risk
        self.gameplay_risk = gameplay_risk

        # Create players with profiles
        for i, (name, profile_type) in enumerate(player_config):
            profile = self._create_profile(profile_type, i)
            player = Player(name, i, profile)
            self.players.append(player)

    def _create_profile(self, profile_type: str, player_id: int) -> PlayerProfile:
        """
        Create a player profile based on type.

        Parameters
        ----------
        profile_type : str
            Type of profile: "human", "simple", "ai", "random", "ml"
        player_id : int
            ID of the player (for ML profile game state access).

        Returns
        -------
        PlayerProfile
            The created profile.
        """
        if profile_type == "human":
            return HumanProfile()
        elif profile_type == "simple" or profile_type == "heuristic":
            return HeuristicPlayer()
        elif profile_type == "ai":
            return AIPlayer(self.ai_decision_maker)
        elif profile_type == "random":
            return RandomPlayer()
        elif profile_type == "ml" or profile_type == "ml_sklearn":
            return self._create_ml_player_profile(player_id, backend="supervised")
        elif profile_type == "ml_rl":
            return self._create_ml_player_profile(player_id, backend="rl")
        elif profile_type == "ml_gan":
            return self._create_ml_player_profile(player_id, backend="gan")
        elif profile_type == "ml_pytorch":
            return self._create_ml_profile(player_id)
        else:
            raise ValueError(f"Unknown profile type: {profile_type}")

    def _create_ml_profile(self, player_id: int) -> MLBasedProfile:
        """
        Create an ML-based profile.

        Parameters
        ----------
        player_id : int
            ID of the player.

        Returns
        -------
        MLBasedProfile
            The ML profile.
        """
        # Lazy initialization of ML components
        if self._ml_model is None:
            config = MLConfig()
            self._ml_model = EuchreMLModel(config)
            self._ml_encoder = GameStateEncoder()

            # Try to load existing weights
            trump_path = config.get_model_path("order_up.pth")
            card_path = config.get_model_path("play_card.pth")
            discard_path = config.get_model_path("discard.pth")

            if trump_path.exists() or card_path.exists() or discard_path.exists():
                self._ml_model.load_weights(
                    trump_path=str(trump_path) if trump_path.exists() else None,
                    card_play_path=str(card_path) if card_path.exists() else None,
                    discard_path=str(discard_path) if discard_path.exists() else None,
                )

        # Create game state provider function
        def get_game_state() -> tuple[int, int, int]:
            """Get current game state for ML profile."""
            trick_num = getattr(self, "_current_trick_number", 0)
            tricks_won = getattr(self, "_current_tricks_won", [0, 0])
            return (trick_num, tricks_won[0], tricks_won[1])

        return MLBasedProfile(
            self._ml_model,
            self._ml_encoder,
            get_game_state,
            trump_selection_risk=self.trump_selection_risk,
            gameplay_risk=self.gameplay_risk,
        )

    def _create_ml_player_profile(self, player_id: int, backend: str = "supervised") -> MLPlayer:
        """
        Create an ML player profile.

        Parameters
        ----------
        player_id : int
            ID of the player.
        backend : str
            ML backend: "supervised", "gan", or "rl"

        Returns
        -------
        MLPlayer
            The ML player profile.
        """
        # Create game state provider function
        def get_game_state() -> tuple[int, int, int]:
            """Get current game state for ML player."""
            trick_num = getattr(self, "_current_trick_number", 0)
            tricks_won = getattr(self, "_current_tricks_won", [0, 0])
            return (trick_num, tricks_won[0], tricks_won[1])

        return MLPlayer(
            backend=backend,
            model_type="random_forest",
            game_state_provider=get_game_state,
            trump_selection_risk=self.trump_selection_risk,
            gameplay_risk=self.gameplay_risk,
        )

    def set_tui(self, tui) -> None:
        """
        Set the TUI object for human players.

        Parameters
        ----------
        tui
            The TUI object.
        """
        # Set players list in TUI for table display
        if hasattr(tui, "set_players"):
            tui.set_players(self.players, self.dealer_id)
        self.tui = tui

        for player in self.players:
            if isinstance(player.profile, HumanProfile):
                player.profile.set_tui(tui)

    def play_hand(self) -> bool:
        """
        Play a single hand.

        Returns
        -------
        bool
            True if game should continue, False if game is over.
        """
        # Reset trick winner for new hand (first trick always starts with player left of dealer)
        if hasattr(self, "_last_trick_winner"):
            delattr(self, "_last_trick_winner")

        # Reset game state tracking for ML profiles
        self._current_trick_number = 0
        self._current_tricks_won = [0, 0]

        # Start new hand log
        if self.tui is not None and hasattr(self.tui, "start_new_hand"):
            self.tui.start_new_hand()

        # Update TUI with initial scores
        if self.tui is not None and hasattr(self.tui, "display_scores"):
            scores = self.get_scores()
            self.tui.display_scores(scores[0], scores[1])

        # Deal cards
        deck = Deck()
        deck.shuffle()

        # Deal 5 cards to each player
        for player in self.players:
            cards = deck.deal(5)
            player.receive_hand(cards)

        # Log initial hands
        if self.tui is not None and hasattr(self.tui, "log_initial_hands"):
            self.tui.log_initial_hands(self.players)

        # Turn up one card
        self.turned_card = deck.draw_one()

        # Log turned card
        if self.tui is not None and hasattr(self.tui, "log_turned_card"):
            self.tui.log_turned_card(self.turned_card)

        # Display turned card prominently after deal
        if self.tui is not None and hasattr(self.tui, "display_turned_card"):
            self.tui.display_turned_card(self.turned_card)

        # Select trump
        self.trump_selector = TrumpSelector(self.players, self.tui)
        self.trump_suit = self.trump_selector.select_trump(self.turned_card, self.dealer_id)

        # Log full trump-decision history into TUI for easier post-hand inspection
        if self.tui is not None and hasattr(self.tui, "log_trump_decision_history"):
            history = self.trump_selector.get_decision_history()
            self.tui.log_trump_decision_history(history.get("order_up", []), history.get("call_trump", []))

        # If all passed, redeal
        if self.trump_suit is None:
            return True  # Continue game, redeal

        # Update TUI with dealer info
        if self.tui is not None and hasattr(self.tui, "dealer_id"):
            self.tui.dealer_id = self.dealer_id

        # Play 5 tricks
        tricks_won = [0, 0]  # Team 0 and Team 1
        for trick_num in range(5):
            self._current_trick_number = trick_num
            # Update TUI with trick number
            if self.tui is not None and hasattr(self.tui, "update_trick_number"):
                self.tui.update_trick_number(trick_num + 1)
            # Note: Scores are displayed after hand completes, not during each trick

            winner_id = self._play_trick()
            winner = self.players[winner_id]
            tricks_won[winner.team] += 1
            self._current_tricks_won = tricks_won.copy()

            # Display trick winner and wait for spacebar
            if self.tui is not None and hasattr(self.tui, "display_trick_winner"):
                self.tui.display_trick_winner(winner.name)

            # Log trick
            if self.tui is not None and hasattr(self.tui, "log_trick"):
                if hasattr(self, "_last_trick_cards") and hasattr(self, "_last_trick_player_ids"):
                    self.tui.log_trick(
                        trick_num + 1,
                        self._last_trick_cards,
                        self._last_trick_player_ids,
                        winner_id,
                    )

        # Score the hand
        self._score_hand(tricks_won)

        # Log hand score
        if self.tui is not None and hasattr(self.tui, "log_hand_score"):
            scores = self.get_scores()
            self.tui.log_hand_score(tricks_won, scores)

        # Display hand log
        if self.tui is not None and hasattr(self.tui, "display_hand_log"):
            self.tui.display_hand_log()

        # Rotate dealer
        self.dealer_id = (self.dealer_id + 1) % 4

        # Check for game end
        return not self._is_game_over()

    def _play_trick(self) -> int:
        """
        Play a single trick.

        Returns
        -------
        int
            ID of the winning player.
        """
        # Determine who leads
        if not hasattr(self, "_last_trick_winner"):
            # First trick: player left of dealer leads
            leader_id = (self.dealer_id + 1) % 4
        else:
            # Subsequent tricks: last trick winner leads
            leader_id = self._last_trick_winner

        played_cards: List[Card] = []
        player_ids: List[int] = []
        led_suit: Optional[Suit] = None

        # Reset trick state in TUI
        if self.tui is not None and hasattr(self.tui, "update_trick_state"):
            self.tui.update_trick_state([], [])

        # Each player plays a card
        for i in range(4):
            player_idx = (leader_id + i) % 4
            player = self.players[player_idx]

            # Get card to play
            card = player.play_card(led_suit, self.trump_suit, played_cards, player_ids)

            # Validate play
            if not self.rules.can_play_card(card, player.hand, led_suit, self.trump_suit):
                raise ValueError(f"Invalid card play: {card}")

            # Play the card
            player.remove_card(card)
            played_cards.append(card)
            player_ids.append(player_idx)

            # Update TUI with current trick state
            if self.tui is not None and hasattr(self.tui, "update_trick_state"):
                self.tui.update_trick_state(played_cards, player_ids)

            # Set led suit if first card
            if i == 0:
                led_suit = self._get_card_suit_for_led(card)

        # Store trick info for logging
        self._last_trick_cards = played_cards.copy()
        self._last_trick_player_ids = player_ids.copy()

        # Determine winner
        winner_id = self.rules.determine_trick_winner(
            played_cards, player_ids, led_suit, self.trump_suit
        )
        self._last_trick_winner = winner_id
        return winner_id

    def _get_card_suit_for_led(self, card: Card) -> Suit:
        """
        Get the suit that a card represents when leading.

        Parameters
        ----------
        card : Card
            The card that was led.

        Returns
        -------
        Suit
            The suit for following purposes.
        """
        if self.trump_suit is None:
            return card.suit

        # Right Bower leads as trump
        if card.rank.value == 11 and card.suit == self.trump_suit:
            return self.trump_suit

        # Left Bower leads as trump
        if card.rank.value == 11:
            trump_card = Card(self.trump_suit, card.rank)
            if card.is_same_color(trump_card):
                return self.trump_suit

        # Regular cards lead as their suit
        return card.suit

    def _score_hand(self, tricks_won: List[int]) -> None:
        """
        Score a completed hand.

        Parameters
        ----------
        tricks_won : List[int]
            Number of tricks won by each team [team0, team1].
        """
        # Determine which team made trump
        # Track which player/team called trump
        if self.trump_selector is None:
            return

        # Find which player called trump (simplified - check first team)
        # In a full implementation, we'd track this during trump selection
        making_team = 0  # Simplified - would track actual trump maker
        defending_team = 1

        making_tricks = tricks_won[making_team]

        if making_tricks >= 5:
            # March: 2 points
            self.scores[making_team] += 2
        elif making_tricks >= 3:
            # 3-4 tricks: 1 point
            self.scores[making_team] += 1
        else:
            # Euchred: defending team gets 2 points
            self.scores[defending_team] += 2

    def _is_game_over(self) -> bool:
        """
        Check if the game is over (a team has 10+ points).

        Returns
        -------
        bool
            True if game is over, False otherwise.
        """
        return self.scores[0] >= 10 or self.scores[1] >= 10

    def get_winner(self) -> Optional[int]:
        """
        Get the winning team ID.

        Returns
        -------
        Optional[int]
            Team ID (0 or 1) if game is over, None otherwise.
        """
        if not self._is_game_over():
            return None
        if self.scores[0] >= 10:
            return 0
        return 1

    def get_scores(self) -> Tuple[int, int]:
        """
        Get current team scores.

        Returns
        -------
        Tuple[int, int]
            (Team 0 score, Team 1 score).
        """
        return (self.scores[0], self.scores[1])

