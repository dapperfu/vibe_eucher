"""Trick simulator for Stage 1 training."""

import random
from typing import Dict, List, Optional, Tuple

import torch

from eucher.cards import Card, Deck, Rank, Suit
from ..action_space import ActionEncoder, ACTION_SPACE_SIZE
from ..config import ReinforcementEucherConfig
from ..rewards.reward_calculator import RewardCalculator
from ..state_encoder import StateEncoder


class TrickScenario:
    """Represents a trick scenario for training.

    Parameters
    ----------
    player_hand : List[Card]
        Player's hand.
    trump_suit : Optional[Suit]
        Trump suit.
    trick_cards : List[Card]
        Cards already played in trick.
    leader_id : int
        Player who led the trick.
    dealer_id : int
        Dealer ID.
    """

    def __init__(
        self,
        player_hand: List[Card],
        trump_suit: Optional[Suit],
        trick_cards: List[Card],
        leader_id: int,
        dealer_id: int,
    ) -> None:
        """Initialize trick scenario.

        Parameters
        ----------
        player_hand : List[Card]
            Player's hand.
        trump_suit : Optional[Suit]
            Trump suit.
        trick_cards : List[Card]
            Cards already played in trick.
        leader_id : int
            Player who led the trick.
        dealer_id : int
            Dealer ID.
        """
        self.player_hand = player_hand
        self.trump_suit = trump_suit
        self.trick_cards = trick_cards
        self.leader_id = leader_id
        self.dealer_id = dealer_id


class TrickSimulator:
    """Simulator for generating random trick scenarios.

    Parameters
    ----------
    config : ReinforcementEucherConfig
        Configuration.
    """

    def __init__(self, config: ReinforcementEucherConfig) -> None:
        """Initialize trick simulator.

        Parameters
        ----------
        config : ReinforcementEucherConfig
            Configuration.
        """
        self.config = config
        self.state_encoder = StateEncoder()
        self.reward_calculator = RewardCalculator(config)
        self.deck = Deck()

    def generate_scenario(self, player_position: int = 0) -> TrickScenario:
        """Generate a random trick scenario.

        Parameters
        ----------
        player_position : int
            Position of the player (0-3).

        Returns
        -------
        TrickScenario
            Random trick scenario.
        """
        # Create a fresh deck
        deck = Deck()
        deck.shuffle()

        # Deal 5 cards to player
        player_hand = deck.deal(5)

        # Random trump suit
        trump_suit = random.choice(list(Suit)) if random.random() > 0.2 else None

        # Random trick cards (0-3 cards already played)
        num_trick_cards = random.randint(0, 3)
        trick_cards = []
        remaining_cards = [card for card in deck.cards if card not in player_hand]
        for _ in range(num_trick_cards):
            if remaining_cards:
                card = random.choice(remaining_cards)
                trick_cards.append(card)
                remaining_cards.remove(card)

        # Random leader and dealer
        leader_id = random.randint(0, 3)
        dealer_id = random.randint(0, 3)

        return TrickScenario(
            player_hand=player_hand,
            trump_suit=trump_suit,
            trick_cards=trick_cards,
            leader_id=leader_id,
            dealer_id=dealer_id,
        )

    def get_legal_actions(self, scenario: TrickScenario) -> List[int]:
        """Get legal actions for scenario.

        Parameters
        ----------
        scenario : TrickScenario
            Trick scenario.

        Returns
        -------
        List[int]
            List of legal action IDs.
        """
        legal_actions = []

        # If trick is empty, can play any card
        if not scenario.trick_cards:
            for i in range(len(scenario.player_hand)):
                if i < 5:
                    legal_actions.append(ActionEncoder.encode_play_action(i))
        else:
            # Must follow suit if possible
            lead_card = scenario.trick_cards[0]
            lead_suit = lead_card.suit

            # Check if player can follow suit
            can_follow = any(
                card.suit == lead_suit
                or (scenario.trump_suit and card.suit == scenario.trump_suit and card.rank == Rank.JACK)
                or (scenario.trump_suit and lead_suit == scenario.trump_suit and card.rank == Rank.JACK)
                for card in scenario.player_hand
            )

            if can_follow:
                # Can only play cards of lead suit or trump
                for i, card in enumerate(scenario.player_hand):
                    if i < 5:
                        if (
                            card.suit == lead_suit
                            or (scenario.trump_suit and card.suit == scenario.trump_suit)
                            or (scenario.trump_suit and lead_suit == scenario.trump_suit and card.rank == Rank.JACK)
                        ):
                            legal_actions.append(ActionEncoder.encode_play_action(i))
            else:
                # Can play any card
                for i in range(len(scenario.player_hand)):
                    if i < 5:
                        legal_actions.append(ActionEncoder.encode_play_action(i))

        return legal_actions

    def simulate_trick_outcome(
        self,
        scenario: TrickScenario,
        played_card: Card,
    ) -> Tuple[bool, float]:
        """Simulate trick outcome.

        Parameters
        ----------
        scenario : TrickScenario
            Trick scenario.
        played_card : Card
            Card played by player.

        Returns
        -------
        Tuple[bool, float]
            (won_trick, reward).
        """
        # Simple heuristic: compare card strength
        # In real game, would need to simulate other players' cards
        # For training, use simplified logic

        all_trick_cards = scenario.trick_cards + [played_card]

        # Determine winner (simplified)
        # In real implementation, would need to consider:
        # - Trump suit
        # - Lead suit
        # - Card ranks
        # - Opponent cards

        # For now, use random outcome weighted by card strength
        card_strength = self._get_card_strength(played_card, scenario.trump_suit)
        win_probability = min(0.8, 0.3 + card_strength * 0.5)

        won_trick = random.random() < win_probability
        reward = self.reward_calculator.calculate_trick_reward(won_trick)

        return won_trick, reward

    def _get_card_strength(self, card: Card, trump_suit: Optional[Suit]) -> float:
        """Get card strength (0.0-1.0).

        Parameters
        ----------
        card : Card
            Card to evaluate.
        trump_suit : Optional[Suit]
            Trump suit.

        Returns
        -------
        float
            Card strength.
        """
        # Right bower (trump jack) is strongest
        if trump_suit and card.suit == trump_suit and card.rank == Rank.JACK:
            return 1.0

        # Left bower (other suit jack when trump is set)
        if trump_suit and card.rank == Rank.JACK:
            # Check if this is the left bower
            for suit in Suit:
                if suit != trump_suit:
                    if card.suit == suit:
                        return 0.9

        # Trump cards
        if trump_suit and card.suit == trump_suit:
            rank_values = {
                Rank.ACE: 0.8,
                Rank.KING: 0.7,
                Rank.QUEEN: 0.6,
                Rank.TEN: 0.5,
                Rank.NINE: 0.4,
            }
            return rank_values.get(card.rank, 0.3)

        # Non-trump cards
        rank_values = {
            Rank.ACE: 0.6,
            Rank.KING: 0.5,
            Rank.QUEEN: 0.4,
            Rank.JACK: 0.3,
            Rank.TEN: 0.2,
            Rank.NINE: 0.1,
        }
        return rank_values.get(card.rank, 0.0)

    def create_training_example(
        self,
        scenario: TrickScenario,
        action: int,
        reward: float,
    ) -> Dict:
        """Create training example from scenario.

        Parameters
        ----------
        scenario : TrickScenario
            Trick scenario.
        action : int
            Action taken.
        reward : float
            Reward received.

        Returns
        -------
        Dict
            Training example.
        """
        # Create mock game state for encoding
        # In real implementation, would use actual game object
        # For now, create minimal state representation
        legal_actions = self.get_legal_actions(scenario)

        # Encode state (simplified - would need full game object)
        # For trick-only training, we can use a simplified encoding
        state_tensor = self._encode_trick_state(scenario, legal_actions)

        return {
            "state": state_tensor,
            "action": action,
            "reward": reward,
            "legal_mask": torch.tensor([1.0 if i in legal_actions else 0.0 for i in range(ACTION_SPACE_SIZE)]),
        }

    def _encode_trick_state(self, scenario: TrickScenario, legal_actions: List[int]) -> torch.Tensor:
        """Encode trick state (full dimension to match model).

        Parameters
        ----------
        scenario : TrickScenario
            Trick scenario.
        legal_actions : List[int]
            Legal actions.

        Returns
        -------
        torch.Tensor
            Encoded state (295 dims to match full state encoder).
        """
        # Full encoding to match model expectations (295 dims)
        # Hand encoding (24 dims)
        hand_encoding = self.state_encoder.encode_hand(scenario.player_hand)

        # Trump encoding (5 dims)
        trump_encoding = self.state_encoder.encode_trump(scenario.trump_suit)

        # Position encodings (8 dims)
        dealer_encoding = self.state_encoder.encode_position(scenario.dealer_id)
        leader_encoding = self.state_encoder.encode_position(scenario.leader_id)

        # Trick cards encoding (96 dims: 24 × 4)
        trick_tensor = torch.zeros(24, 4)
        for i, card in enumerate(scenario.trick_cards[:4]):
            idx = self.state_encoder._card_to_index(card)
            trick_tensor[idx, i] = 1.0
        trick_encoding = trick_tensor.flatten()

        # Trick history encoding (120 dims: 24 × 5) - empty for trick-only
        history_encoding = torch.zeros(24, 5).flatten()

        # Unseen cards encoding (24 dims) - simplified: mark cards not in hand as unseen
        unseen_encoding = torch.ones(24)
        for card in scenario.player_hand:
            idx = self.state_encoder._card_to_index(card)
            unseen_encoding[idx] = 0.0
        for card in scenario.trick_cards:
            idx = self.state_encoder._card_to_index(card)
            unseen_encoding[idx] = 0.0

        # Legal actions (18 dims)
        legal_mask = torch.tensor([1.0 if i in legal_actions else 0.0 for i in range(ACTION_SPACE_SIZE)])

        # Concatenate to match full state encoder (295 dims total)
        state = torch.cat([
            hand_encoding,  # 24
            trump_encoding,  # 5
            dealer_encoding,  # 4
            leader_encoding,  # 4
            trick_encoding,  # 96
            history_encoding,  # 120
            unseen_encoding,  # 24
            legal_mask,  # 18
        ])

        return state

