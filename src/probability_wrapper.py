"""Wrapper class that displays probability distributions for player decisions.

This module provides a ProbabilityAwareProfile wrapper that intercepts decision
calls, extracts and displays probability distributions, then returns the original
decision.
"""

from typing import TYPE_CHECKING, List, Optional

from src.cards import Card, Suit
from src.decision_probability_extractor import (
    get_call_trump_probabilities,
    get_discard_probabilities,
    get_order_up_probabilities,
    get_play_card_probabilities,
)
from src.player_profiles import PlayerProfile

if TYPE_CHECKING:
    from src.players import Player


class ProbabilityAwareProfile(PlayerProfile):
    """Wrapper profile that displays probabilities before making decisions.

    This wrapper intercepts decision calls, extracts probability distributions,
    displays them, and then returns the original decision from the wrapped profile.

    Parameters
    ----------
    wrapped_profile : PlayerProfile
        The original profile to wrap.
    display_fn : callable, optional
        Function to call for displaying probabilities. Should accept (player_name, decision_type, probabilities, selected).
        If None, uses default print-based display.
    """

    def __init__(self, wrapped_profile: PlayerProfile, display_fn=None) -> None:
        """Initialize probability-aware profile wrapper.

        Parameters
        ----------
        wrapped_profile : PlayerProfile
            The original profile to wrap.
        display_fn : callable, optional
            Function to call for displaying probabilities.
            Should accept (player_name, decision_type, probabilities, selected).
            If None, uses default print-based display.
        """
        self.wrapped_profile = wrapped_profile
        self.display_fn = display_fn or self._default_display

    def _default_display(
        self, player_name: str, decision_type: str, probabilities: dict, selected
    ) -> None:
        """
        Default display function for probabilities.

        Parameters
        ----------
        player_name : str
            Name of the player making the decision.
        decision_type : str
            Type of decision (e.g., "Order Up", "Call Trump", "Discard", "Play Card").
        probabilities : dict
            Dictionary mapping options to probabilities.
        selected
            The selected option.
        """
        print(f"\n{player_name}: {decision_type}")
        # Sort by probability (descending) for better readability
        sorted_items = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

        for option, prob in sorted_items:
            marker = " (selected)" if option == selected else ""
            if isinstance(option, bool):
                option_str = "Order up" if option else "Pass"
            elif option is None:
                option_str = "Pass"
            elif isinstance(option, Suit):
                option_str = f"{option.value}"
            elif isinstance(option, Card):
                option_str = str(option)
            else:
                option_str = str(option)

            print(f"  {option_str}: {prob:.3f}{marker}")

    def decide_order_up(
        self, player: "Player", turned_card: Card, dealer_id: int, trump_suit: Optional[Suit]
    ) -> bool:
        """
        Decide whether to order up, displaying probabilities first.

        Parameters
        ----------
        player : Player
            The player making the decision.
        turned_card : Card
            The card that was turned up.
        dealer_id : int
            The ID of the dealer.
        trump_suit : Optional[Suit]
            Current trump suit if already determined.

        Returns
        -------
        bool
            True to order up, False to pass.
        """
        # Extract probabilities
        probabilities = get_order_up_probabilities(
            self.wrapped_profile, player, turned_card, dealer_id, trump_suit
        )

        # Make decision
        decision = self.wrapped_profile.decide_order_up(player, turned_card, dealer_id, trump_suit)

        # Display probabilities
        self.display_fn(player.name, "Order Up", probabilities, decision)

        return decision

    def decide_call_trump(
        self,
        player: "Player",
        turned_card: Card,
        trump_suit: Optional[Suit],
        must_choose: bool = False,
    ) -> Optional[Suit]:
        """
        Decide which suit to call as trump, displaying probabilities first.

        Parameters
        ----------
        player : Player
            The player making the decision.
        turned_card : Card
            The card that was turned up (cannot be chosen).
        trump_suit : Optional[Suit]
            Current trump suit if already determined.
        must_choose : bool
            If True, must choose a suit.

        Returns
        -------
        Optional[Suit]
            The suit to call as trump, or None to pass.
        """
        # Extract probabilities
        probabilities = get_call_trump_probabilities(
            self.wrapped_profile, player, turned_card, trump_suit, must_choose
        )

        # Make decision
        decision = self.wrapped_profile.decide_call_trump(player, turned_card, trump_suit, must_choose)

        # Display probabilities
        self.display_fn(player.name, "Call Trump", probabilities, decision)

        return decision

    def choose_card_to_discard(
        self, player: "Player", turned_card: Optional[Card] = None, ordered_up_by: Optional[str] = None
    ) -> Card:
        """
        Choose a card to discard, displaying probabilities first.

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
        # Extract probabilities
        probabilities = get_discard_probabilities(
            self.wrapped_profile, player, turned_card, ordered_up_by
        )

        # Make decision
        decision = self.wrapped_profile.choose_card_to_discard(player, turned_card, ordered_up_by)

        # Display probabilities
        self.display_fn(player.name, "Discard", probabilities, decision)

        return decision

    def play_card(
        self,
        player: "Player",
        led_suit: Optional[Suit],
        trump_suit: Optional[Suit],
        trick_cards: List[Card],
        trick_player_ids: List[int],
    ) -> Card:
        """
        Choose a card to play, displaying probabilities first.

        Parameters
        ----------
        player : Player
            The player making the decision.
        led_suit : Optional[Suit]
            The suit that was led, if any.
        trump_suit : Optional[Suit]
            The current trump suit, if any.
        trick_cards : List[Card]
            Cards already played in the trick.
        trick_player_ids : List[int]
            Player IDs who played each card in trick_cards.

        Returns
        -------
        Card
            The card to play.
        """
        # Extract probabilities
        probabilities = get_play_card_probabilities(
            self.wrapped_profile, player, led_suit, trump_suit, trick_cards, trick_player_ids
        )

        # Make decision
        decision = self.wrapped_profile.play_card(
            player, led_suit, trump_suit, trick_cards, trick_player_ids
        )

        # Display probabilities
        self.display_fn(player.name, "Play Card", probabilities, decision)

        return decision

    # Delegate other methods to wrapped profile
    def __getattr__(self, name: str):
        """Delegate attribute access to wrapped profile."""
        return getattr(self.wrapped_profile, name)





