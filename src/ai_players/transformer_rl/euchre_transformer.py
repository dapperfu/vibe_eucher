"""Transformer-based neural network architecture for Euchre RL.

This module implements a transformer encoder with multi-head attention for
processing game state and outputting action distributions and value estimates.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from eucher.cards import Suit


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer sequences.

    Parameters
    ----------
    d_model : int
        Model dimension (embedding size).
    max_len : int
        Maximum sequence length.
    """

    def __init__(self, d_model: int, max_len: int = 100) -> None:
        """Initialize positional encoding.

        Parameters
        ----------
        d_model : int
            Model dimension.
        max_len : int
            Maximum sequence length.
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, d_model).

        Returns
        -------
        torch.Tensor
            Input with positional encoding added.
        """
        return x + self.pe[:, : x.size(1), :]


class CardEmbedding(nn.Module):
    """Embedding layer for card features.

    Parameters
    ----------
    input_dim : int
        Input feature dimension per card.
    embedding_dim : int
        Output embedding dimension.
    """

    def __init__(self, input_dim: int = 12, embedding_dim: int = 64) -> None:
        """Initialize card embedding.

        Parameters
        ----------
        input_dim : int
            Input feature dimension.
        embedding_dim : int
            Embedding dimension.
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., input_dim).

        Returns
        -------
        torch.Tensor
            Embedded tensor of shape (..., embedding_dim).
        """
        return self.norm(F.relu(self.linear(x)))


class EuchreTransformer(nn.Module):
    """Transformer-based network for Euchre RL with multi-head outputs.

    This network processes game state through a transformer encoder and outputs:
    - Bidding actions (order up, call trump)
    - Discard actions
    - Card play actions
    - Auxiliary predictions (expected tricks, win probability, partner trump probability)
    - Value estimate (for actor-critic)

    Parameters
    ----------
    card_embedding_dim : int
        Dimension of card embeddings.
    d_model : int
        Transformer model dimension.
    nhead : int
        Number of attention heads.
    num_layers : int
        Number of transformer encoder layers.
    dim_feedforward : int
        Feedforward network dimension.
    dropout : float
        Dropout probability.
    max_hand_size : int
        Maximum hand size (6 cards).
    num_suits : int
        Number of suits (4).
    """

    def __init__(
        self,
        card_embedding_dim: int = 64,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_hand_size: int = 6,
        num_suits: int = 4,
    ) -> None:
        """Initialize transformer network.

        Parameters
        ----------
        card_embedding_dim : int
            Card embedding dimension.
        d_model : int
            Transformer model dimension.
        nhead : int
            Number of attention heads.
        num_layers : int
            Number of transformer layers.
        dim_feedforward : int
            Feedforward dimension.
        dropout : float
            Dropout probability.
        max_hand_size : int
            Maximum hand size.
        num_suits : int
            Number of suits.
        """
        super().__init__()

        self.card_embedding_dim = card_embedding_dim
        self.d_model = d_model
        self.max_hand_size = max_hand_size
        self.num_suits = num_suits

        # Card embedding
        card_input_dim = 12  # suit(4) + rank(6) + is_trump(1) + trump_rank(1)
        self.card_embedding = CardEmbedding(card_input_dim, card_embedding_dim)

        # Project card embeddings to model dimension
        self.card_projection = nn.Linear(card_embedding_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=200)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Context feature dimensions
        # Hand: max_hand_size cards
        # Trick history: up to 5 tricks × 4 cards = 20 cards
        # Current trick: up to 4 cards
        # Deduction maps: 24 cards × 4 players = 96 cards
        # Game context: ~20 features
        # Total sequence length: ~140 tokens

        # Shared feature processing
        self.shared_mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Output heads
        # Bidding head: order_up (binary) + call_trump (5: 4 suits + pass)
        self.order_up_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

        self.call_trump_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 5),  # 4 suits + pass
        )

        # Discard head: 6 card positions
        self.discard_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, max_hand_size),
        )

        # Play card head: 6 card positions
        self.play_card_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, max_hand_size),
        )

        # Auxiliary heads
        self.expected_tricks_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

        self.win_probability_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),  # Probability output
        )

        self.partner_trump_prob_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),  # Probability output
        )

        # Value head (for critic)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

        # Context projection: game_context is typically 13 features
        context_dim = 13  # From encode_game_context
        self.context_projection = nn.Linear(context_dim, d_model)

    def forward(
        self,
        hand: torch.Tensor,
        trick_history: torch.Tensor,
        game_context: torch.Tensor,
        deduction_maps: Optional[torch.Tensor] = None,
        current_trick: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through transformer network.

        Parameters
        ----------
        hand : torch.Tensor
            Hand encoding of shape (batch, max_hand_size, card_feature_dim).
        trick_history : torch.Tensor
            Trick history encoding of shape (batch, num_tricks, 4, card_feature_dim).
        game_context : torch.Tensor
            Game context features of shape (batch, context_dim).
        deduction_maps : Optional[torch.Tensor]
            Deduction probability maps of shape (batch, 24, 4) for 24 cards × 4 players.
        current_trick : Optional[torch.Tensor]
            Current trick cards of shape (batch, 4, card_feature_dim).
        attention_mask : Optional[torch.Tensor]
            Attention mask of shape (batch, seq_len) to mask out padding.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - 'order_up': logits for order up decision (batch, 1)
            - 'call_trump': logits for trump selection (batch, 5)
            - 'discard': logits for discard selection (batch, max_hand_size)
            - 'play_card': logits for card play (batch, max_hand_size)
            - 'expected_tricks': expected tricks prediction (batch, 1)
            - 'win_probability': win probability (batch, 1)
            - 'partner_trump_prob': partner trump probability (batch, 1)
            - 'value': state value estimate (batch, 1)
        """
        batch_size = hand.shape[0]

        # Embed and project all card sequences
        hand_embedded = self.card_embedding(hand)  # (batch, max_hand_size, card_embedding_dim)
        hand_projected = self.card_projection(hand_embedded)  # (batch, max_hand_size, d_model)

        # Process trick history
        trick_history_flat = trick_history.view(batch_size, -1, hand.shape[-1])  # (batch, num_tricks*4, card_feat)
        trick_history_embedded = self.card_embedding(trick_history_flat)
        trick_history_projected = self.card_projection(trick_history_embedded)

        # Process current trick if provided
        current_trick_projected = None
        if current_trick is not None:
            # Ensure current_trick has batch dimension
            if current_trick.dim() == 2:
                # Add batch dimension if needed: (seq_len, card_feat) -> (1, seq_len, card_feat)
                current_trick = current_trick.unsqueeze(0)
            elif current_trick.dim() == 3:
                # Already has batch dimension: (batch, seq_len, card_feat)
                pass
            else:
                # Unexpected shape, skip current trick
                current_trick = None
            
            if current_trick is not None:
                current_trick_embedded = self.card_embedding(current_trick)
                current_trick_projected = self.card_projection(current_trick_embedded)

        # Process deduction maps if provided
        deduction_projected = None
        if deduction_maps is not None:
            # Ensure deduction_maps has batch dimension
            if deduction_maps.dim() == 2:
                # Add batch dimension: (24, 4) -> (1, 24, 4)
                deduction_maps = deduction_maps.unsqueeze(0)
            elif deduction_maps.dim() == 3:
                # Already has batch dimension: (batch, 24, 4)
                pass
            else:
                # Unexpected shape, skip deduction maps
                deduction_maps = None
            
            if deduction_maps is not None:
                # Ensure deduction_maps batch size matches hand batch size
                if deduction_maps.shape[0] != batch_size:
                    if deduction_maps.shape[0] == 1:
                        # Expand single batch to match
                        deduction_maps = deduction_maps.expand(batch_size, -1, -1)
                    else:
                        # Take first batch_size elements
                        deduction_maps = deduction_maps[:batch_size]
                
                # Reshape deduction maps: (batch, 24, 4) -> (batch, 24*4, 1) -> expand to card features
                # For simplicity, we'll treat each card-player pair as a feature
                # In practice, we'd need to encode this more carefully
                deduction_flat = deduction_maps.view(batch_size, -1, 1)  # (batch, 96, 1)
                # Expand to match card feature dimension (simplified)
                deduction_expanded = deduction_flat.expand(-1, -1, self.card_embedding_dim)
                deduction_embedded = self.card_embedding(
                    torch.zeros(batch_size, deduction_flat.shape[1], 12, device=hand.device)
                )  # Placeholder - would need proper encoding
                deduction_projected = self.card_projection(deduction_embedded)

        # Concatenate all sequences
        # Ensure all parts have the same batch size
        sequence_parts = [hand_projected, trick_history_projected]
        if current_trick_projected is not None:
            # Ensure batch size matches
            if current_trick_projected.shape[0] != batch_size:
                if current_trick_projected.shape[0] == 1 and batch_size > 1:
                    current_trick_projected = current_trick_projected.expand(batch_size, -1, -1)
                elif current_trick_projected.shape[0] > batch_size:
                    current_trick_projected = current_trick_projected[:batch_size]
            sequence_parts.append(current_trick_projected)
        if deduction_projected is not None:
            # Ensure batch size matches
            if deduction_projected.shape[0] != batch_size:
                if deduction_projected.shape[0] == 1 and batch_size > 1:
                    deduction_projected = deduction_projected.expand(batch_size, -1, -1)
                elif deduction_projected.shape[0] > batch_size:
                    deduction_projected = deduction_projected[:batch_size]
            sequence_parts.append(deduction_projected)

        # Add game context as a special token
        # game_context is (batch, context_dim), need to project to (batch, 1, d_model)
        if game_context.dim() == 2:
            # Ensure batch size matches
            if game_context.shape[0] != batch_size:
                if game_context.shape[0] == 1 and batch_size > 1:
                    game_context = game_context.expand(batch_size, -1)
                elif game_context.shape[0] > batch_size:
                    game_context = game_context[:batch_size]
            # Expand to (batch, 1, context_dim) then project
            context_expanded = game_context.unsqueeze(1)  # (batch, 1, context_dim)
            context_projected = self.context_projection(context_expanded)  # (batch, 1, d_model)
        else:
            # Already 3D, ensure batch size matches
            if game_context.shape[0] != batch_size:
                if game_context.shape[0] == 1 and batch_size > 1:
                    game_context = game_context.expand(batch_size, -1, -1)
                elif game_context.shape[0] > batch_size:
                    game_context = game_context[:batch_size]
            # Project if needed
            if game_context.shape[-1] != self.d_model:
                context_projected = self.context_projection(game_context)
            else:
                context_projected = game_context
        sequence_parts.append(context_projected)

        # Concatenate all parts - all should now have shape (batch_size, seq_len, d_model)
        full_sequence = torch.cat(sequence_parts, dim=1)  # (batch, seq_len, d_model)

        # Add positional encoding
        full_sequence = self.pos_encoder(full_sequence)

        # Create attention mask if not provided
        if attention_mask is None:
            seq_len = full_sequence.shape[1]
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=hand.device)

        # Transformer encoding
        # Convert boolean mask to float mask for transformer
        # True values become 0.0 (attend), False become -inf (mask)
        float_mask = attention_mask.float()
        float_mask = float_mask.masked_fill(float_mask == 0, float("-inf"))
        float_mask = float_mask.masked_fill(float_mask == 1, 0.0)

        transformer_out = self.transformer_encoder(full_sequence, src_key_padding_mask=~attention_mask)

        # Use CLS token (first token) or mean pooling for global representation
        # For now, use mean pooling over sequence
        global_repr = transformer_out.mean(dim=1)  # (batch, d_model)

        # Process through shared MLP
        shared_features = self.shared_mlp(global_repr)  # (batch, d_model)

        # Generate outputs from all heads
        outputs = {
            "order_up": self.order_up_head(shared_features),
            "call_trump": self.call_trump_head(shared_features),
            "discard": self.discard_head(shared_features),
            "play_card": self.play_card_head(shared_features),
            "expected_tricks": self.expected_tricks_head(shared_features),
            "win_probability": self.win_probability_head(shared_features),
            "partner_trump_prob": self.partner_trump_prob_head(shared_features),
            "value": self.value_head(shared_features),
        }

        return outputs

    def get_parameter_count(self) -> int:
        """Get total number of trainable parameters.

        Returns
        -------
        int
            Number of parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_transformer_network(
    card_embedding_dim: int = 64,
    d_model: int = 256,
    nhead: int = 8,
    num_layers: int = 4,
    dim_feedforward: int = 512,
    dropout: float = 0.1,
    device: Optional[torch.device] = None,
) -> EuchreTransformer:
    """Create and initialize a transformer network.

    Parameters
    ----------
    card_embedding_dim : int
        Card embedding dimension.
    d_model : int
        Transformer model dimension.
    nhead : int
        Number of attention heads.
    num_layers : int
        Number of transformer layers.
    dim_feedforward : int
        Feedforward dimension.
    dropout : float
        Dropout probability.
    device : Optional[torch.device]
        Device to create network on.

    Returns
    -------
    EuchreTransformer
        Initialized transformer network.
    """
    network = EuchreTransformer(
        card_embedding_dim=card_embedding_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )

    if device is not None:
        network = network.to(device)

    return network

