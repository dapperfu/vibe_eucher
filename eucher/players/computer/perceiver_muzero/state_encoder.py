"""Token-based state encoder for EuchrePerceiverMuZero Perceiver-IO."""

from typing import Dict, List, Optional

import torch

from eucher.cards import Card, Rank, Suit
from eucher.game import Game

from .config import Mode, Phase, PerceiverMuZeroConfig


class StateEncoder:
    """Encode game state as tokens for Perceiver-IO input."""

    NUM_CARDS = 24
    NUM_SUITS = 4
    NUM_RANKS = 6
    TOKEN_DIM = 64  # Token embedding dimension

    def __init__(self, config: PerceiverMuZeroConfig) -> None:
        """
        Initialize state encoder.

        Parameters
        ----------
        config : PerceiverMuZeroConfig
            Configuration object.
        """
        self.config = config
        self.token_dim = config.token_dim

    def encode_tokens(self, game: Game, player_id: int) -> torch.Tensor:
        """
        Encode game state as variable-length token sequence.

        Parameters
        ----------
        game : Game
            The game instance.
        player_id : int
            ID of the player (0-3).

        Returns
        -------
        torch.Tensor
            Token sequence tensor [num_tokens, token_dim].
        """
        tokens: List[torch.Tensor] = []

        player = game.players[player_id]

        # Card tokens: hand (5 tokens)
        for card in player.hand[:5]:
            tokens.append(self._encode_card_token(card))

        # Card tokens: played cards (up to 20 cards, 5 tricks × 4 players)
        for trick in getattr(game, "tricks", [])[:5]:
            if trick:
                for card in trick:
                    if card:
                        tokens.append(self._encode_card_token(card))

        # Card token: kitty (turned card)
        if game.turned_card:
            tokens.append(self._encode_card_token(game.turned_card))

        # Trick history tokens (one per trick)
        num_tricks = len(getattr(game, "tricks", []))
        for i in range(min(5, num_tricks)):
            tokens.append(self._encode_trick_token(game, i))

        # Bidding sequence tokens
        bidding_tokens = self._encode_bidding_sequence(game, player_id)
        tokens.extend(bidding_tokens)

        # Phase token
        tokens.append(self._encode_phase_token(game))

        # Mode token
        tokens.append(self._encode_mode_token(game))

        # Dealer token
        tokens.append(self._encode_dealer_token(game, player_id))

        # Variant flags tokens
        tokens.extend(self._encode_variant_flags())

        # Legal action mask tokens
        legal_actions = self._get_legal_actions(game, player_id)
        tokens.append(self._encode_legal_action_mask(legal_actions))

        # Belief state tokens (unseen cards probability, suit voids, opponent modeling)
        belief_tokens = self._encode_belief_state(game, player_id)
        tokens.extend(belief_tokens)

        # Stack all tokens
        if tokens:
            return torch.stack(tokens)
        else:
            # Return empty token sequence with correct dimension
            return torch.zeros(0, self.token_dim)

    def _encode_card_token(self, card: Card) -> torch.Tensor:
        """
        Encode a card as a token.

        Parameters
        ----------
        card : Card
            Card to encode.

        Returns
        -------
        torch.Tensor
            Card token [token_dim].
        """
        token = torch.zeros(self.token_dim)

        # Card index (0-23)
        card_idx = self._card_to_index(card)
        token[0:24] = torch.zeros(24)
        token[card_idx] = 1.0

        # Suit one-hot (24-27)
        suit_idx = list(Suit).index(card.suit)
        token[24 + suit_idx] = 1.0

        # Rank one-hot (28-33)
        ranks = [Rank.NINE, Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE]
        rank_idx = ranks.index(card.rank)
        token[28 + rank_idx] = 1.0

        # Is trump (34)
        # This will be set based on game state, default to 0
        token[34] = 0.0

        # Is left bower (35)
        token[35] = 0.0

        # Is right bower (36)
        token[36] = 0.0

        # Remaining dimensions for additional features
        # (37-63): reserved for future use

        return token

    def _encode_trick_token(self, game: Game, trick_index: int) -> torch.Tensor:
        """
        Encode a trick as a token.

        Parameters
        ----------
        game : Game
            Game instance.
        trick_index : int
            Index of trick (0-4).

        Returns
        -------
        torch.Tensor
            Trick token [token_dim].
        """
        token = torch.zeros(self.token_dim)

        # Trick index (0-4)
        token[0] = float(trick_index)

        # Number of cards played in trick (1-4)
        tricks = getattr(game, "tricks", [])
        if trick_index < len(tricks):
            trick = tricks[trick_index]
            num_cards = len([c for c in trick if c is not None])
            token[1] = float(num_cards)
        else:
            token[1] = 0.0

        # Winner of trick (2-5: one-hot for player 0-3)
        # Simplified: would need to track actual winner
        token[2:6] = torch.zeros(4)

        # Remaining dimensions reserved
        return token

    def _encode_bidding_sequence(self, game: Game, player_id: int) -> List[torch.Tensor]:
        """
        Encode bidding sequence as tokens.

        Parameters
        ----------
        game : Game
            Game instance.
        player_id : int
            Player ID.

        Returns
        -------
        List[torch.Tensor]
            List of bidding tokens.
        """
        tokens: List[torch.Tensor] = []

        # Simplified: encode current bidding state
        # In full implementation, would track full sequence

        # Bidding phase indicator
        token = torch.zeros(self.token_dim)
        if game.trump_suit is None:
            token[0] = 1.0  # Still bidding
        else:
            token[1] = 1.0  # Bidding complete

        # Current bidder (if applicable)
        token[2:6] = torch.zeros(4)

        tokens.append(token)

        return tokens

    def _encode_phase_token(self, game: Game) -> torch.Tensor:
        """
        Encode game phase as token.

        Parameters
        ----------
        game : Game
            Game instance.

        Returns
        -------
        torch.Tensor
            Phase token [token_dim].
        """
        token = torch.zeros(self.token_dim)

        # Determine phase
        if game.trump_suit is None:
            if game.turned_card:
                phase = Phase.PICKUP
            else:
                phase = Phase.DEAL
        else:
            # Check if still playing tricks
            tricks = getattr(game, "tricks", [])
            if len(tricks) < 5:
                phase = Phase.PLAY
            else:
                phase = Phase.SCORE

        # Phase one-hot encoding (0-4)
        phase_map = {
            Phase.DEAL: 0,
            Phase.PICKUP: 1,
            Phase.CALL_TRUMP: 2,
            Phase.PLAY: 3,
            Phase.SCORE: 4,
        }
        phase_idx = phase_map[phase]
        token[phase_idx] = 1.0

        return token

    def _encode_mode_token(self, game: Game) -> torch.Tensor:
        """
        Encode game mode as token.

        Parameters
        ----------
        game : Game
            Game instance.

        Returns
        -------
        torch.Tensor
            Mode token [token_dim].
        """
        token = torch.zeros(self.token_dim)

        # Check if alone mode
        is_alone = getattr(game, "alone_mode", False)
        if is_alone:
            token[0] = 1.0  # ALONE mode
        else:
            token[1] = 1.0  # NORMAL mode

        return token

    def _encode_dealer_token(self, game: Game, player_id: int) -> torch.Tensor:
        """
        Encode dealer position as token.

        Parameters
        ----------
        game : Game
            Game instance.
        player_id : int
            Player ID.

        Returns
        -------
        torch.Tensor
            Dealer token [token_dim].
        """
        token = torch.zeros(self.token_dim)

        # Dealer position (0-3)
        dealer_pos = (game.dealer_id - player_id) % 4
        token[0:4] = torch.zeros(4)
        token[dealer_pos] = 1.0

        # Is player dealer
        token[4] = 1.0 if game.dealer_id == player_id else 0.0

        return token

    def _encode_variant_flags(self) -> List[torch.Tensor]:
        """
        Encode variant flags as tokens.

        Returns
        -------
        List[torch.Tensor]
            List of variant flag tokens.
        """
        tokens: List[torch.Tensor] = []

        # Screw the dealer flag
        token1 = torch.zeros(self.token_dim)
        token1[0] = 1.0 if self.config.enable_screw_the_dealer else 0.0
        tokens.append(token1)

        # 9-10 trade-in flag
        token2 = torch.zeros(self.token_dim)
        token2[0] = 1.0 if self.config.enable_nine_ten_tradein else 0.0
        tokens.append(token2)

        # Go alone flag
        token3 = torch.zeros(self.token_dim)
        token3[0] = 1.0 if self.config.enable_go_alone else 0.0
        tokens.append(token3)

        return tokens

    def _encode_legal_action_mask(self, legal_actions: List[int]) -> torch.Tensor:
        """
        Encode legal action mask as token.

        Parameters
        ----------
        legal_actions : List[int]
            List of legal action IDs.

        Returns
        -------
        torch.Tensor
            Legal action mask token [token_dim].
        """
        token = torch.zeros(self.token_dim)

        # Encode legal actions (up to action_space_size)
        for action_id in legal_actions:
            if action_id < self.token_dim:
                token[action_id] = 1.0

        return token

    def _encode_belief_state(self, game: Game, player_id: int) -> List[torch.Tensor]:
        """
        Encode belief state tokens (unseen cards, suit voids, opponent modeling).

        Parameters
        ----------
        game : Game
            Game instance.
        player_id : int
            Player ID.

        Returns
        -------
        List[torch.Tensor]
            List of belief state tokens.
        """
        tokens: List[torch.Tensor] = []

        # Unseen cards probability distribution (simplified)
        # One token per opponent (3 opponents)
        for opp_id in range(4):
            if opp_id != player_id:
                token = torch.zeros(self.token_dim)
                # Encode probability that opponent has each card
                # Simplified: uniform distribution
                for i in range(min(24, self.token_dim)):
                    token[i] = 1.0 / 24.0
                tokens.append(token)

        # Suit void inference tokens (one per suit)
        for suit_idx in range(4):
            token = torch.zeros(self.token_dim)
            # Encode void probabilities for each opponent
            # Simplified: no void inference initially
            token[0:3] = torch.zeros(3)
            tokens.append(token)

        return tokens

    def _get_legal_actions(self, game: Game, player_id: int) -> List[int]:
        """
        Get list of legal action IDs for current state.

        Parameters
        ----------
        game : Game
            Game instance.
        player_id : int
            Player ID.

        Returns
        -------
        List[int]
            List of legal action IDs.
        """
        # Simplified: return all actions (illegal plays allowed per spec)
        # In full implementation, would filter based on game state
        from .action_space import ACTION_SPACE_SIZE

        return list(range(ACTION_SPACE_SIZE))

    def _card_to_index(self, card: Card) -> int:
        """
        Convert card to index (0-23).

        Parameters
        ----------
        card : Card
            Card to convert.

        Returns
        -------
        int
            Card index (0-23).
        """
        suits = list(Suit)
        ranks = [Rank.NINE, Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE]
        suit_idx = suits.index(card.suit)
        rank_idx = ranks.index(card.rank)
        return suit_idx * 6 + rank_idx

