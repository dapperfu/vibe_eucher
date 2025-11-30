"""Assistant helper for displaying bot decision recommendations."""

from typing import Dict, List, Optional, Tuple

import numpy as np

from eucher.cards import Card, Rank, Suit
from eucher.game import Game
from eucher.plugins import get_registry
from eucher.players.base import Player, PlayerProfile
from eucher.players.profiles import HumanProfile
from eucher.rules import RulesEngine


class AssistantHelper:
    """Helper class for getting bot recommendations for decisions."""

    def __init__(self, bot_type: str, game: Optional[Game] = None) -> None:
        """
        Initialize the assistant helper.

        Parameters
        ----------
        bot_type : str
            Type of bot to use as assistant (e.g., "random", "heuristic", "ai").
        game : Optional[Game]
            Game instance (required for some bot types that need game state).
        """
        self.bot_type = bot_type.lower()
        self.game = game
        self.rules = RulesEngine()
        self._profile: Optional[PlayerProfile] = None
        self._dummy_player: Optional[Player] = None

    def _get_profile(self) -> PlayerProfile:
        """
        Get or create the bot profile.

        Returns
        -------
        PlayerProfile
            The bot profile instance.
        """
        if self._profile is None:
            # Special case: human profile (not allowed as assistant)
            if self.bot_type == "human":
                raise ValueError("Human profile cannot be used as assistant")

            # Try to find plugin in registry
            registry = get_registry()
            plugin_metadata = registry.get(self.bot_type)

            if plugin_metadata is not None:
                # Found plugin - use it
                kwargs: dict = {}
                
                # Add player_id for ML players that need it
                if plugin_metadata.requires_game or "ml" in self.bot_type.lower():
                    kwargs["player_id"] = 0
                
                # Create profile using plugin factory
                if plugin_metadata.requires_game:
                    if self.game is None:
                        raise ValueError(f"Bot type '{self.bot_type}' requires a game instance")
                    kwargs["game"] = self.game
                
                factory = plugin_metadata.factory
                self._profile = factory(**kwargs)
            else:
                raise ValueError(f"Unknown bot type: {self.bot_type}")

        return self._profile

    def _get_dummy_player(self, hand: List[Card]) -> Player:
        """
        Get or create a dummy player with the given hand.

        Parameters
        ----------
        hand : List[Card]
            Hand to give to the dummy player.

        Returns
        -------
        Player
            Dummy player instance.
        """
        if self._dummy_player is None:
            profile = self._get_profile()
            self._dummy_player = Player("Assistant", 0, profile)
        
        # Update hand
        self._dummy_player.hand = hand.copy()
        return self._dummy_player

    def _calculate_hand_strength(self, hand: List[Card], trump_suit: Suit) -> float:
        """
        Calculate hand strength for a given trump suit.

        Parameters
        ----------
        hand : List[Card]
            Player's hand.
        trump_suit : Suit
            The trump suit to evaluate against.

        Returns
        -------
        float
            Hand strength score (higher is better).
        """
        total_power = 0.0
        trump_count = 0
        has_bower = False

        for card in hand:
            # Calculate card power
            if card.rank == Rank.JACK:
                if card.suit == trump_suit:
                    # Right bower
                    total_power += 100.0
                    has_bower = True
                    trump_count += 1
                else:
                    # Check if left bower
                    trump_card = Card(trump_suit, Rank.ACE)
                    if card.is_same_color(trump_card):
                        total_power += 90.0
                        has_bower = True
                        trump_count += 1
                    else:
                        total_power += 20.0
            elif card.suit == trump_suit:
                # Trump card
                trump_count += 1
                if card.rank == Rank.ACE:
                    total_power += 80.0
                elif card.rank == Rank.KING:
                    total_power += 70.0
                elif card.rank == Rank.QUEEN:
                    total_power += 60.0
                elif card.rank == Rank.TEN:
                    total_power += 50.0
                elif card.rank == Rank.NINE:
                    total_power += 40.0
            else:
                # Off-suit card
                if card.rank == Rank.ACE:
                    total_power += 35.0
                elif card.rank == Rank.KING:
                    total_power += 30.0
                elif card.rank == Rank.QUEEN:
                    total_power += 25.0
                elif card.rank == Rank.JACK:
                    total_power += 20.0
                elif card.rank == Rank.TEN:
                    total_power += 15.0
                elif card.rank == Rank.NINE:
                    total_power += 10.0

        # Bonus for having bowers
        if has_bower:
            total_power += 20.0

        # Bonus for multiple trump cards
        if trump_count >= 3:
            total_power += 15.0
        elif trump_count >= 2:
            total_power += 10.0

        return total_power

    def get_order_up_recommendation(
        self, hand: List[Card], turned_card: Card, dealer_id: int
    ) -> Tuple[Dict[bool, float], Optional[float]]:
        """
        Get recommendation probabilities for order up decision.

        Parameters
        ----------
        hand : List[Card]
            Player's hand.
        turned_card : Card
            Card that was turned up.
        dealer_id : int
            ID of the dealer.

        Returns
        -------
        Tuple[Dict[bool, float], Optional[float]]
            Dictionary mapping decision (True/False) to probability, and hand strength.
        """
        hand_strength = None
        try:
            profile = self._get_profile()
            dummy_player = self._get_dummy_player(hand)
            
            # Calculate hand strength for the turned card's suit
            hand_strength = self._calculate_hand_strength(hand, turned_card.suit)
            
            if self.bot_type == "random":
                # Random bot: equal probability for both options
                return {True: 0.5, False: 0.5}, hand_strength
            
            # For EuchreZero, try to get full policy distribution
            if self.bot_type == "euchre_zero" and hasattr(profile, 'mcts'):
                try:
                    from eucher.players.computer.euchre_zero.action_space import ActionEncoder
                    import torch
                    
                    # Encode state
                    state_dict = profile.state_encoder.encode_full_state(
                        profile._get_game(dummy_player), dummy_player.player_id
                    )
                    state_dict["risk_factor"] = torch.tensor([profile.risk_factor])
                    state_tensor = profile.state_encoder.encode_state_dict_to_tensor(state_dict)
                    
                    # Run MCTS to get policy
                    policy = profile.mcts.search(state_tensor, profile.risk_factor)
                    
                    # Extract order up probabilities
                    # ACTION_SPACE: PASS=0, ORDER_UP=1
                    pass_prob = float(policy[0])
                    order_up_prob = float(policy[1])
                    
                    # Normalize
                    total = pass_prob + order_up_prob
                    if total > 0:
                        pass_prob /= total
                        order_up_prob /= total
                    
                    return {False: pass_prob, True: order_up_prob}, hand_strength
                except Exception:
                    # Fallback to decision-based approach
                    pass
            
            # For other bots, determine what they would choose
            decision = profile.decide_order_up(dummy_player, turned_card, dealer_id, None)
            return {decision: 1.0, not decision: 0.0}, hand_strength
        except Exception:
            # Fallback: equal distribution
            return {True: 0.5, False: 0.5}, hand_strength

    def get_call_trump_recommendation(
        self, hand: List[Card], turned_card: Card, must_choose: bool = False
    ) -> Tuple[Dict[Optional[Suit], float], Optional[Dict[Suit, float]]]:
        """
        Get recommendation probabilities for call trump decision.

        Parameters
        ----------
        hand : List[Card]
            Player's hand.
        turned_card : Card
            Card that was turned up (cannot be chosen).
        must_choose : bool
            Whether a suit must be chosen (cannot pass).

        Returns
        -------
        Tuple[Dict[Optional[Suit], float], Optional[Dict[Suit, float]]]
            Dictionary mapping suit (or None for pass) to probability, and hand strengths for each suit.
        """
        forbidden_suit = turned_card.suit
        available_suits = [suit for suit in Suit if suit != forbidden_suit]
        hand_strengths: Optional[Dict[Suit, float]] = {}
        
        # Calculate hand strength for each available suit
        for suit in available_suits:
            hand_strengths[suit] = self._calculate_hand_strength(hand, suit)
        
        try:
            profile = self._get_profile()
            dummy_player = self._get_dummy_player(hand)
            
            if self.bot_type == "random":
                # Random bot: equal probability for all options
                if must_choose:
                    # Must choose a suit
                    prob = 1.0 / len(available_suits)
                    return {suit: prob for suit in available_suits}, hand_strengths
                else:
                    # Can pass or choose a suit
                    num_options = len(available_suits) + 1
                    prob = 1.0 / num_options
                    result = {None: prob}
                    for suit in available_suits:
                        result[suit] = prob
                    return result, hand_strengths
            
            # For EuchreZero, try to get full policy distribution
            if self.bot_type == "euchre_zero" and hasattr(profile, 'mcts'):
                try:
                    from eucher.players.computer.euchre_zero.action_space import ActionEncoder
                    import torch
                    
                    # Encode state
                    state_dict = profile.state_encoder.encode_full_state(
                        profile._get_game(dummy_player), dummy_player.player_id
                    )
                    state_dict["risk_factor"] = torch.tensor([profile.risk_factor])
                    state_tensor = profile.state_encoder.encode_state_dict_to_tensor(state_dict)
                    
                    # Run MCTS to get policy
                    policy = profile.mcts.search(state_tensor, profile.risk_factor)
                    
                    # Extract call trump probabilities
                    # ACTION_SPACE: PASS=0, ORDER_UP=1, CALL_HEARTS=2, CALL_DIAMONDS=3, CALL_CLUBS=4, CALL_SPADES=5
                    pass_prob = float(policy[0])
                    call_probs = {
                        Suit.HEARTS: float(policy[2]),
                        Suit.DIAMONDS: float(policy[3]),
                        Suit.CLUBS: float(policy[4]),
                        Suit.SPADES: float(policy[5]),
                    }
                    
                    # Remove forbidden suit
                    if forbidden_suit in call_probs:
                        del call_probs[forbidden_suit]
                    
                    # Normalize
                    total = pass_prob + sum(call_probs.values())
                    if total > 0:
                        pass_prob /= total
                        for suit in call_probs:
                            call_probs[suit] /= total
                    
                    result: Dict[Optional[Suit], float] = {}
                    for suit in available_suits:
                        result[suit] = call_probs.get(suit, 0.0)
                    if not must_choose:
                        result[None] = pass_prob
                    
                    return result, hand_strengths
                except Exception:
                    # Fallback to decision-based approach
                    pass
            
            # For other bots, determine what they would choose
            decision = profile.decide_call_trump(dummy_player, turned_card, None, must_choose)
            result = {}
            for suit in available_suits:
                result[suit] = 0.0
            if not must_choose:
                result[None] = 0.0
            result[decision] = 1.0
            return result, hand_strengths
        except Exception:
            # Fallback: equal distribution
            if must_choose:
                prob = 1.0 / len(available_suits)
                return {suit: prob for suit in available_suits}, hand_strengths
            else:
                num_options = len(available_suits) + 1
                prob = 1.0 / num_options
                result = {None: prob}
                for suit in available_suits:
                    result[suit] = prob
                return result, hand_strengths

    def get_play_card_recommendation(
        self,
        hand: List[Card],
        valid_cards: List[Card],
        led_suit: Optional[Suit],
        trump_suit: Optional[Suit],
        trick_cards: List[Card],
        trick_player_ids: List[int],
    ) -> Dict[Card, float]:
        """
        Get recommendation probabilities for card play decision.

        Parameters
        ----------
        hand : List[Card]
            Player's hand.
        valid_cards : List[Card]
            Valid cards that can be played.
        led_suit : Optional[Suit]
            Suit that was led.
        trump_suit : Optional[Suit]
            Current trump suit.
        trick_cards : List[Card]
            Cards already played in the trick.
        trick_player_ids : List[int]
            Player IDs who played each card in trick_cards.

        Returns
        -------
        Dict[Card, float]
            Dictionary mapping card to probability.
        """
        try:
            profile = self._get_profile()
            dummy_player = self._get_dummy_player(hand)
            
            if self.bot_type == "random":
                # Random bot: equal probability for all valid cards
                prob = 1.0 / len(valid_cards) if valid_cards else 0.0
                return {card: prob for card in valid_cards}
            
            # For EuchreZero, try to get full policy distribution
            if self.bot_type == "euchre_zero" and hasattr(profile, 'mcts') and trump_suit is not None:
                try:
                    import torch
                    
                    # Encode state
                    state_dict = profile.state_encoder.encode_full_state(
                        profile._get_game(dummy_player), dummy_player.player_id
                    )
                    state_dict["risk_factor"] = torch.tensor([profile.risk_factor])
                    state_tensor = profile.state_encoder.encode_state_dict_to_tensor(state_dict)
                    
                    # Run MCTS to get policy
                    policy = profile.mcts.search(state_tensor, profile.risk_factor)
                    
                    # Extract play card probabilities
                    # ACTION_SPACE: PLAY_0=13, PLAY_1=14, PLAY_2=15, PLAY_3=16, PLAY_4=17
                    play_probs = policy[13:18]
                    
                    # Map to actual cards in hand
                    result: Dict[Card, float] = {}
                    for card in valid_cards:
                        result[card] = 0.0
                    
                    # Distribute probabilities to valid cards
                    # Note: policy indices correspond to hand positions
                    total_prob = 0.0
                    for i, card in enumerate(hand):
                        if i < len(play_probs) and card in valid_cards:
                            prob = float(play_probs[i])
                            result[card] = prob
                            total_prob += prob
                    
                    # Normalize
                    if total_prob > 0:
                        for card in result:
                            result[card] /= total_prob
                    
                    return result
                except Exception:
                    # Fallback to decision-based approach
                    pass
            
            # For other bots, determine what they would choose
            decision = profile.play_card(
                dummy_player, led_suit, trump_suit, trick_cards, trick_player_ids
            )
            
            # Initialize result with all valid cards at 0.0
            result: Dict[Card, float] = {}
            for card in valid_cards:
                result[card] = 0.0
            
            # Find matching card in valid_cards (by suit and rank, not object identity)
            matching_card = None
            for card in valid_cards:
                if card.suit == decision.suit and card.rank == decision.rank:
                    matching_card = card
                    break
            
            # If we found a match, set its probability to 1.0
            if matching_card is not None:
                result[matching_card] = 1.0
            else:
                # If decision card is not in valid_cards, distribute equally
                # This shouldn't happen, but handle gracefully
                prob = 1.0 / len(valid_cards) if valid_cards else 0.0
                for card in valid_cards:
                    result[card] = prob
            
            return result
        except Exception:
            # Fallback: equal distribution
            prob = 1.0 / len(valid_cards) if valid_cards else 0.0
            return {card: prob for card in valid_cards}

    def get_discard_recommendation(
        self, hand: List[Card], turned_card: Optional[Card] = None, ordered_up_by: Optional[str] = None
    ) -> Dict[Card, float]:
        """
        Get recommendation probabilities for discard decision.

        Parameters
        ----------
        hand : List[Card]
            Player's hand (should have 6 cards).
        turned_card : Optional[Card]
            Card that was ordered up, if available.
        ordered_up_by : Optional[str]
            Name of the player who ordered up, if available.

        Returns
        -------
        Dict[Card, float]
            Dictionary mapping card to probability.
        """
        try:
            profile = self._get_profile()
            dummy_player = self._get_dummy_player(hand)
            
            if self.bot_type == "random":
                # Random bot: equal probability for all cards
                prob = 1.0 / len(hand) if hand else 0.0
                return {card: prob for card in hand}
            
            # For EuchreZero, try to get full policy distribution
            if self.bot_type == "euchre_zero" and hasattr(profile, 'mcts'):
                try:
                    import torch
                    
                    # Encode state
                    state_dict = profile.state_encoder.encode_full_state(
                        profile._get_game(dummy_player), dummy_player.player_id
                    )
                    state_dict["risk_factor"] = torch.tensor([profile.risk_factor])
                    state_tensor = profile.state_encoder.encode_state_dict_to_tensor(state_dict)
                    
                    # Run MCTS to get policy
                    policy = profile.mcts.search(state_tensor, profile.risk_factor)
                    
                    # Extract discard probabilities
                    # ACTION_SPACE: DISCARD_0=7, DISCARD_1=8, DISCARD_2=9, DISCARD_3=10, DISCARD_4=11, DISCARD_5=12
                    discard_probs = policy[7:13]
                    
                    # Map to actual cards in hand
                    result: Dict[Card, float] = {}
                    total_prob = 0.0
                    for i, card in enumerate(hand):
                        if i < len(discard_probs):
                            prob = float(discard_probs[i])
                            result[card] = prob
                            total_prob += prob
                    
                    # Normalize
                    if total_prob > 0:
                        for card in result:
                            result[card] /= total_prob
                    
                    return result
                except Exception:
                    # Fallback to decision-based approach
                    pass
            
            # For other bots, determine what they would choose
            decision = profile.choose_card_to_discard(dummy_player, turned_card, ordered_up_by)
            
            # Initialize result with all cards at 0.0
            result: Dict[Card, float] = {}
            for card in hand:
                result[card] = 0.0
            
            # Find matching card in hand (by suit and rank, not object identity)
            matching_card = None
            for card in hand:
                if card.suit == decision.suit and card.rank == decision.rank:
                    matching_card = card
                    break
            
            # If we found a match, set its probability to 1.0
            if matching_card is not None:
                result[matching_card] = 1.0
            else:
                # If decision card is not in hand, distribute equally
                # This shouldn't happen, but handle gracefully
                prob = 1.0 / len(hand) if hand else 0.0
                for card in hand:
                    result[card] = prob
            
            return result
        except Exception:
            # Fallback: equal distribution
            prob = 1.0 / len(hand) if hand else 0.0
            return {card: prob for card in hand}

