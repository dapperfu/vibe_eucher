"""Perceiver-IO encoder for EucherPerceiverMuZero."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import PerceiverMuZeroConfig


class PerceiverEncoder(nn.Module):
    """Perceiver-IO encoder for processing variable-length token sequences."""

    def __init__(self, config: PerceiverMuZeroConfig) -> None:
        """
        Initialize Perceiver-IO encoder.

        Parameters
        ----------
        config : PerceiverMuZeroConfig
            Configuration object.
        """
        super().__init__()
        self.config = config
        self.token_dim = config.token_dim
        self.latent_size = config.latent_size
        self.latent_slots = config.latent_slots

        # Input token embedding
        self.input_embedding = nn.Linear(self.token_dim, self.latent_size)

        # Latent array initialization
        self.latent_array = nn.Parameter(
            torch.randn(self.latent_slots, self.latent_size) * 0.02
        )

        # Cross-attention: input tokens attend to latent array
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.latent_size,
            num_heads=8,
            batch_first=True,
        )
        self.cross_attention_norm = nn.LayerNorm(self.latent_size)

        # Self-attention: latent array self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.latent_size,
            num_heads=8,
            batch_first=True,
        )
        self.self_attention_norm = nn.LayerNorm(self.latent_size)

        # Feed-forward network for latent
        self.ffn = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size * 4),
            nn.GELU(),
            nn.Linear(self.latent_size * 4, self.latent_size),
        )
        self.ffn_norm = nn.LayerNorm(self.latent_size)

        # Output projection to fixed-size latent vector
        self.output_projection = nn.Linear(self.latent_size, self.latent_size)

    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Perceiver-IO encoder.

        Parameters
        ----------
        input_tokens : torch.Tensor
            Input token sequence [batch_size, num_tokens, token_dim] or [num_tokens, token_dim].

        Returns
        -------
        torch.Tensor
            Fixed-size latent vector [batch_size, latent_size] or [latent_size].
        """
        # Handle single sample (no batch dimension)
        single_sample = False
        if input_tokens.dim() == 2:
            single_sample = True
            input_tokens = input_tokens.unsqueeze(0)  # Add batch dimension

        batch_size = input_tokens.size(0)
        num_tokens = input_tokens.size(1)

        # Embed input tokens
        embedded_tokens = self.input_embedding(input_tokens)  # [batch, num_tokens, latent_size]

        # Initialize latent array (broadcast to batch size)
        latent = self.latent_array.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [batch, latent_slots, latent_size]

        # Cross-attention: latent queries attend to input tokens
        # Query: latent array, Key/Value: input tokens
        latent_attended, _ = self.cross_attention(
            query=latent, key=embedded_tokens, value=embedded_tokens
        )
        latent = self.cross_attention_norm(latent + latent_attended)

        # Self-attention: latent array self-attention
        latent_self_attended, _ = self.self_attention(
            query=latent, key=latent, value=latent
        )
        latent = self.self_attention_norm(latent + latent_self_attended)

        # Feed-forward network
        latent_ffn = self.ffn(latent)
        latent = self.ffn_norm(latent + latent_ffn)

        # Pool latent array to fixed-size vector (mean pooling)
        latent_pooled = latent.mean(dim=1)  # [batch, latent_size]

        # Output projection
        output = self.output_projection(latent_pooled)  # [batch, latent_size]

        if single_sample:
            output = output.squeeze(0)  # Remove batch dimension

        return output

