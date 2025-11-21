"""Main game logic and state management for Euchre."""

from typing import List, Optional, Tuple

from src.ai import AIDecisionMaker
from src.cards import Card, Deck, Suit
from src.player_profiles import AIBasedProfile, HumanProfile, PlayerProfile, SimpleRuleBasedProfile
from src.players import Player
from src.rules import RulesEngine
from src.trump import TrumpSelector


class Game:
    """Manages the Euchre game state and flow."""

    def __init__(self, player_config: List[Tuple[str, str]]) -> None:
        """
        Initialize a new game.

        Parameters
        ----------
        player_config : List[Tuple[str, str]]
            List of (name, profile_type) tuples for each player.
            profile_type can be: "human", "simple", "ai"
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

        # Create players with profiles
        for i, (name, profile_type) in enumerate(player_config):
            profile = self._create_profile(profile_type)
            player = Player(name, i, profile)
            self.players.append(player)

    def _create_profile(self, profile_type: str) -> PlayerProfile:
        """
        Create a player profile based on type.

        Parameters
        ----------
        profile_type : str
            Type of profile: "human", "simple", "ai"

        Returns
        -------
        PlayerProfile
            The created profile.
        """
        if profile_type == "human":
            return HumanProfile()
        elif profile_type == "simple":
            return SimpleRuleBasedProfile()
        elif profile_type == "ai":
            return AIBasedProfile(self.ai_decision_maker)
        else:
            raise ValueError(f"Unknown profile type: {profile_type}")

    def set_tui(self, tui) -> None:
        """
        Set the TUI object for human players.

        Parameters
        ----------
        tui
            The TUI object.
        """
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
        # Deal cards
        deck = Deck()
        deck.shuffle()

        # Deal 5 cards to each player
        for player in self.players:
            cards = deck.deal(5)
            player.receive_hand(cards)

        # Turn up one card
        self.turned_card = deck.draw_one()

        # Select trump
        self.trump_selector = TrumpSelector(self.players)
        self.trump_suit = self.trump_selector.select_trump(self.turned_card, self.dealer_id)

        # If all passed, redeal
        if self.trump_suit is None:
            return True  # Continue game, redeal

        # Play 5 tricks
        tricks_won = [0, 0]  # Team 0 and Team 1
        for trick_num in range(5):
            winner_id = self._play_trick()
            winner = self.players[winner_id]
            tricks_won[winner.team] += 1

        # Score the hand
        self._score_hand(tricks_won)

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

        # Each player plays a card
        for i in range(4):
            player_idx = (leader_id + i) % 4
            player = self.players[player_idx]

            # Get card to play
            card = player.play_card(led_suit, self.trump_suit, played_cards)

            # Validate play
            if not self.rules.can_play_card(card, player.hand, led_suit, self.trump_suit):
                raise ValueError(f"Invalid card play: {card}")

            # Play the card
            player.remove_card(card)
            played_cards.append(card)
            player_ids.append(player_idx)

            # Set led suit if first card
            if i == 0:
                led_suit = self._get_card_suit_for_led(card)

        # Determine winner
        winner_id = self.rules.determine_trick_winner(played_cards, player_ids, led_suit, self.trump_suit)
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
        defending_tricks = tricks_won[defending_team]

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

