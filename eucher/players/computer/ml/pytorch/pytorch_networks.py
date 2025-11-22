"""PyTorch neural network architectures for Euchre AI player."""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CardEmbedding(nn.Module):
    """Card embedding layer that converts card features to dense vectors.

    Parameters
    ----------
    input_dim : int
        Input feature dimension (suit + rank + additional features).
    embedding_dim : int
        Output embedding dimension.
    """

    def __init__(self, input_dim: int = 12, embedding_dim: int = 32) -> None:
        """Initialize card embedding layer.

        Parameters
        ----------
        input_dim : int
            Input feature dimension.
        embedding_dim : int
            Embedding dimension.
        """
        super().__init__()
        self.embedding = nn.Linear(input_dim, embedding_dim)
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
        return self.norm(F.relu(self.embedding(x)))


class AttentionMechanism(nn.Module):
    """Attention mechanism for focusing on important cards.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension.
    num_heads : int
        Number of attention heads.
    """

    def __init__(self, embed_dim: int = 32, num_heads: int = 4) -> None:
        """Initialize attention mechanism.

        Parameters
        ----------
        embed_dim : int
            Embedding dimension.
        num_heads : int
            Number of attention heads.
        """
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with self-attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Attended tensor of shape (batch, seq_len, embed_dim).
        """
        attn_out, _ = self.attention(x, x, x)
        return self.norm(x + attn_out)


class HybridNetwork(nn.Module):
    """Hybrid neural network combining CNN, LSTM, and attention mechanisms.

    Parameters
    ----------
    card_embedding_dim : int
        Dimension of card embeddings.
    cnn_channels : int
        Number of CNN channels.
    lstm_hidden_dim : int
        LSTM hidden dimension.
    attention_heads : int
        Number of attention heads.
    hidden_dim : int
        Hidden dimension for MLP layers.
    """

    def __init__(
        self,
        card_embedding_dim: int = 32,
        cnn_channels: int = 64,
        lstm_hidden_dim: int = 128,
        attention_heads: int = 4,
        hidden_dim: int = 256,
    ) -> None:
        """Initialize hybrid network.

        Parameters
        ----------
        card_embedding_dim : int
            Card embedding dimension.
        cnn_channels : int
            CNN channels.
        lstm_hidden_dim : int
            LSTM hidden dimension.
        attention_heads : int
            Attention heads.
        hidden_dim : int
            MLP hidden dimension.
        """
        super().__init__()

        # Card embedding layer
        card_input_dim = 12  # suit(4) + rank(6) + is_trump(1) + trump_rank(1)
        self.card_embedding = CardEmbedding(card_input_dim, card_embedding_dim)

        # CNN branch for hand processing
        self.cnn_branch = nn.Sequential(
            nn.Conv1d(card_embedding_dim, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        # LSTM branch for trick history
        self.lstm_branch = nn.LSTM(
            card_embedding_dim, lstm_hidden_dim, batch_first=True, bidirectional=True
        )

        # Attention mechanism
        self.attention = AttentionMechanism(card_embedding_dim, attention_heads)

        # Feature dimensions
        cnn_output_dim = cnn_channels * 2
        lstm_output_dim = lstm_hidden_dim * 2  # bidirectional
        attention_output_dim = card_embedding_dim * 6  # 6 cards in hand
        context_dim = 13  # Game context features
        summary_dim = 8  # Won tricks summary

        # Combined feature dimension
        combined_dim = (
            cnn_output_dim + lstm_output_dim + attention_output_dim + context_dim + summary_dim
        )

        # Shared MLP layers
        self.shared_mlp = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Decision heads
        # Order up (binary classification)
        self.order_up_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Trump selection (5 classes: 4 suits + pass)
        self.trump_selection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 5),
        )

        # Discard selection (6 classes: one for each card in hand)
        self.discard_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 6),
        )

        # Card play head (variable output, will be filtered by valid plays)
        self.card_play_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 6),  # Max 6 cards in hand
        )

    def forward(
        self,
        hand: torch.Tensor,
        trick_history: torch.Tensor,
        won_tricks_summary: torch.Tensor,
        game_context: torch.Tensor,
        current_trick: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the network.

        Parameters
        ----------
        hand : torch.Tensor
            Hand encoding of shape (batch, 6, card_feature_dim).
        trick_history : torch.Tensor
            Trick history of shape (batch, max_tricks, 4, card_feature_dim).
        won_tricks_summary : torch.Tensor
            Won tricks summary of shape (batch, summary_dim).
        game_context : torch.Tensor
            Game context of shape (batch, context_dim).
        current_trick : Optional[torch.Tensor]
            Current trick cards of shape (batch, 4, card_feature_dim).

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with decision head outputs:
            - 'order_up': logits for order up decision
            - 'trump_selection': logits for trump suit selection
            - 'discard': logits for discard selection
            - 'card_play': logits for card play selection
        """
        batch_size = hand.shape[0]

        # Embed cards
        hand_embedded = self.card_embedding(hand)  # (batch, 6, embed_dim)
        trick_history_flat = trick_history.view(batch_size, -1, hand.shape[-1])  # Flatten tricks
        trick_history_embedded = self.card_embedding(trick_history_flat)  # (batch, seq, embed_dim)

        # CNN branch for hand
        hand_cnn_input = hand_embedded.transpose(1, 2)  # (batch, embed_dim, 6)
        cnn_features = self.cnn_branch(hand_cnn_input)  # (batch, cnn_output_dim)

        # LSTM branch for trick history
        lstm_out, (h_n, c_n) = self.lstm_branch(trick_history_embedded)
        # Use final hidden state
        lstm_features = lstm_out[:, -1, :] if lstm_out.shape[1] > 0 else torch.zeros(
            batch_size, lstm_out.shape[-1], device=hand.device
        )

        # Attention mechanism on hand
        hand_attended = self.attention(hand_embedded)  # (batch, 6, embed_dim)
        attention_features = hand_attended.flatten(1)  # (batch, 6 * embed_dim)

        # Combine all features
        combined_features = torch.cat(
            [cnn_features, lstm_features, attention_features, won_tricks_summary, game_context],
            dim=1,
        )

        # Shared MLP
        shared_features = self.shared_mlp(combined_features)

        # Decision heads
        outputs = {
            "order_up": self.order_up_head(shared_features),
            "trump_selection": self.trump_selection_head(shared_features),
            "discard": self.discard_head(shared_features),
            "card_play": self.card_play_head(shared_features),
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


def create_network(
    card_embedding_dim: int = 32,
    cnn_channels: int = 64,
    lstm_hidden_dim: int = 128,
    attention_heads: int = 4,
    hidden_dim: int = 256,
    device: Optional[torch.device] = None,
) -> HybridNetwork:
    """Create and initialize a hybrid network.

    Parameters
    ----------
    card_embedding_dim : int
        Card embedding dimension.
    cnn_channels : int
        CNN channels.
    lstm_hidden_dim : int
        LSTM hidden dimension.
    attention_heads : int
        Attention heads.
    hidden_dim : int
        MLP hidden dimension.
    device : Optional[torch.device]
        Device to create network on.

    Returns
    -------
    HybridNetwork
        Initialized network.
    """
    network = HybridNetwork(
        card_embedding_dim=card_embedding_dim,
        cnn_channels=cnn_channels,
        lstm_hidden_dim=lstm_hidden_dim,
        attention_heads=attention_heads,
        hidden_dim=hidden_dim,
    )

    if device is not None:
        network = network.to(device)

    return network

