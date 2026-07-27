from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from ..contracts import (
    FusionTransformerOutput,
)


class FusionTransformerLayer(nn.Module):
    """
    One pre-normalization multimodal Transformer encoder layer.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        number_of_attention_heads: int,
        feed_forward_size: int,
        dropout: float,
        attention_dropout: float,
        activation: str,
        layer_norm_epsilon: float,
    ) -> None:
        super().__init__()

        if hidden_size <= 0:
            raise ValueError(
                "Fusion hidden size must be positive."
            )

        if number_of_attention_heads <= 0:
            raise ValueError(
                "The number of attention heads must be positive."
            )

        if hidden_size % number_of_attention_heads != 0:
            raise ValueError(
                "Fusion hidden size must be divisible by the "
                "number of attention heads."
            )

        if feed_forward_size <= 0:
            raise ValueError(
                "Feed-forward size must be positive."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "Fusion dropout must be in the range [0, 1)."
            )

        if not 0.0 <= attention_dropout < 1.0:
            raise ValueError(
                "Attention dropout must be in the range [0, 1)."
            )

        self.hidden_size = hidden_size
        self.number_of_attention_heads = (
            number_of_attention_heads
        )

        self.attention_layer_norm = nn.LayerNorm(
            hidden_size,
            eps=layer_norm_epsilon,
        )

        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=number_of_attention_heads,
            dropout=attention_dropout,
            batch_first=True,
        )

        self.attention_output_dropout = nn.Dropout(
            dropout
        )

        self.feed_forward_layer_norm = nn.LayerNorm(
            hidden_size,
            eps=layer_norm_epsilon,
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(
                hidden_size,
                feed_forward_size,
            ),
            self._build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(
                feed_forward_size,
                hidden_size,
            ),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _build_activation(
        activation: str,
    ) -> nn.Module:
        activations: dict[str, nn.Module] = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
        }

        if activation not in activations:
            raise ValueError(
                f"Unsupported fusion activation '{activation}'. "
                f"Expected one of {sorted(activations)}."
            )

        return activations[activation]

    def forward(
        self,
        hidden_states: Tensor,
        *,
        key_padding_mask: Tensor | None,
        output_attentions: bool,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Apply self-attention and a feed-forward network.

        Args:
            hidden_states:
                [batch_size, sequence_length, hidden_size]

            key_padding_mask:
                Boolean mask with shape:
                [batch_size, sequence_length]

                True means that the position must be ignored.

            output_attentions:
                Whether to return attention probabilities.
        """

        normalized_states = self.attention_layer_norm(
            hidden_states
        )

        attention_output, attention_weights = (
            self.self_attention(
                query=normalized_states,
                key=normalized_states,
                value=normalized_states,
                key_padding_mask=key_padding_mask,
                need_weights=output_attentions,
                average_attn_weights=False,
            )
        )

        hidden_states = (
            hidden_states
            + self.attention_output_dropout(
                attention_output
            )
        )

        normalized_states = self.feed_forward_layer_norm(
            hidden_states
        )

        feed_forward_output = self.feed_forward(
            normalized_states
        )

        hidden_states = (
            hidden_states + feed_forward_output
        )

        if not output_attentions:
            attention_weights = None

        return hidden_states, attention_weights
    
class MultimodalFusionTransformer(nn.Module):
    """
    Transformer encoder that performs multimodal early fusion.

    Expected input layout:
        [MM-CLS] [IMAGE PATCHES] [TEXT TOKENS]
    """

    def __init__(
        self,
        config: dict[str, Any],
    ) -> None:
        super().__init__()

        self.config = config
        self.fusion_config = config["fusion"]

        self.hidden_size: int = self.fusion_config[
            "hidden_size"
        ]

        self.number_of_layers: int = self.fusion_config[
            "number_of_layers"
        ]

        self.number_of_attention_heads: int = (
            self.fusion_config[
                "number_of_attention_heads"
            ]
        )

        self.feed_forward_size: int = self.fusion_config[
            "feed_forward_size"
        ]

        self.dropout_probability: float = (
            self.fusion_config["dropout"]
        )

        self.attention_dropout_probability: float = (
            self.fusion_config["attention_dropout"]
        )

        self.activation_name: str = self.fusion_config[
            "activation"
        ]

        self.layer_norm_epsilon: float = (
            self.fusion_config["layer_norm_epsilon"]
        )

        self.use_final_layer_norm: bool = (
            self.fusion_config["use_final_layer_norm"]
        )

        self.output_hidden_states: bool = (
            self.fusion_config.get(
                "output_hidden_states",
                True,
            )
        )

        self.output_attentions: bool = (
            self.fusion_config.get(
                "output_attentions",
                True,
            )
        )

        self._validate_config(config)

        self.layers = nn.ModuleList(
            [
                FusionTransformerLayer(
                    hidden_size=self.hidden_size,
                    number_of_attention_heads=(
                        self.number_of_attention_heads
                    ),
                    feed_forward_size=(
                        self.feed_forward_size
                    ),
                    dropout=self.dropout_probability,
                    attention_dropout=(
                        self.attention_dropout_probability
                    ),
                    activation=self.activation_name,
                    layer_norm_epsilon=(
                        self.layer_norm_epsilon
                    ),
                )
                for _ in range(self.number_of_layers)
            ]
        )

        if self.use_final_layer_norm:
            self.final_layer_norm: nn.Module = nn.LayerNorm(
                self.hidden_size,
                eps=self.layer_norm_epsilon,
            )
        else:
            self.final_layer_norm = nn.Identity()

    def _validate_config(
        self,
        full_config: dict[str, Any],
    ) -> None:
        required_fields = {
            "hidden_size",
            "number_of_layers",
            "number_of_attention_heads",
            "feed_forward_size",
            "dropout",
            "attention_dropout",
            "activation",
            "layer_norm_epsilon",
            "use_final_layer_norm",
        }

        missing_fields = (
            required_fields - self.fusion_config.keys()
        )

        if missing_fields:
            raise ValueError(
                "Missing fusion configuration fields: "
                f"{sorted(missing_fields)}"
            )

        if self.hidden_size <= 0:
            raise ValueError(
                "Fusion hidden size must be positive."
            )

        if self.number_of_layers <= 0:
            raise ValueError(
                "The fusion Transformer must contain at least "
                "one layer."
            )

        if self.number_of_attention_heads <= 0:
            raise ValueError(
                "The number of fusion attention heads must "
                "be positive."
            )

        if (
            self.hidden_size
            % self.number_of_attention_heads
            != 0
        ):
            raise ValueError(
                "Fusion hidden size must be divisible by the "
                "number of attention heads."
            )

        if self.feed_forward_size <= 0:
            raise ValueError(
                "Fusion feed-forward size must be positive."
            )

        if self.layer_norm_epsilon <= 0:
            raise ValueError(
                "Layer-normalization epsilon must be positive."
            )

        projection_hidden_size = full_config[
            "projection"
        ]["hidden_size"]

        if projection_hidden_size != self.hidden_size:
            raise ValueError(
                "Projection and fusion hidden sizes must match: "
                f"projection={projection_hidden_size}, "
                f"fusion={self.hidden_size}."
            )

    def forward(
        self,
        *,
        embeddings: Tensor,
        attention_mask: Tensor,
        multimodal_cls_index: int | None,
    ) -> FusionTransformerOutput:
        """
        Fuse the multimodal token sequence.

        Args:
            embeddings:
                [batch_size, sequence_length, hidden_size]

            attention_mask:
                [batch_size, sequence_length]

                1 means valid.
                0 means padding.

            multimodal_cls_index:
                Position of the multimodal CLS token.
        """

        self._validate_inputs(
            embeddings=embeddings,
            attention_mask=attention_mask,
            multimodal_cls_index=multimodal_cls_index,
        )

        key_padding_mask = attention_mask.eq(0)

        hidden_states = embeddings

        collected_hidden_states: list[Tensor] | None

        if self.output_hidden_states:
            collected_hidden_states = [
                hidden_states
            ]
        else:
            collected_hidden_states = None

        collected_attentions: list[Tensor] | None

        if self.output_attentions:
            collected_attentions = []
        else:
            collected_attentions = None

        for layer in self.layers:
            hidden_states, attention_weights = layer(
                hidden_states,
                key_padding_mask=key_padding_mask,
                output_attentions=self.output_attentions,
            )

            if collected_hidden_states is not None:
                collected_hidden_states.append(
                    hidden_states
                )

            if (
                collected_attentions is not None
                and attention_weights is not None
            ):
                collected_attentions.append(
                    attention_weights
                )

        hidden_states = self.final_layer_norm(
            hidden_states
        )

        if collected_hidden_states is not None:
            collected_hidden_states[-1] = hidden_states

        cls_embedding = None

        if multimodal_cls_index is not None:
            cls_embedding = hidden_states[
                :,
                multimodal_cls_index,
                :,
            ]

        return FusionTransformerOutput(
            last_hidden_state=hidden_states,
            cls_embedding=cls_embedding,
            attention_mask=attention_mask,
            hidden_states=(
                tuple(collected_hidden_states)
                if collected_hidden_states is not None
                else None
            ),
            attentions=(
                tuple(collected_attentions)
                if collected_attentions is not None
                else None
            ),
        )

    def _validate_inputs(
        self,
        *,
        embeddings: Tensor,
        attention_mask: Tensor,
        multimodal_cls_index: int | None,
    ) -> None:
        if embeddings.ndim != 3:
            raise ValueError(
                "embeddings must have shape "
                "[batch_size, sequence_length, hidden_size]."
            )

        if embeddings.shape[0] <= 0:
            raise ValueError(
                "embeddings must contain at least one sample."
            )

        if embeddings.shape[1] <= 0:
            raise ValueError(
                "embeddings must contain at least one token."
            )

        if embeddings.shape[2] != self.hidden_size:
            raise ValueError(
                "Input hidden size does not match the fusion "
                f"hidden size: expected={self.hidden_size}, "
                f"received={embeddings.shape[2]}."
            )

        if not embeddings.is_floating_point():
            raise TypeError(
                "embeddings must use a floating-point dtype."
            )

        if attention_mask.ndim != 2:
            raise ValueError(
                "attention_mask must have shape "
                "[batch_size, sequence_length]."
            )

        if attention_mask.shape != embeddings.shape[:2]:
            raise ValueError(
                "attention_mask must match the batch and "
                "sequence dimensions of embeddings."
            )

        if attention_mask.device != embeddings.device:
            raise ValueError(
                "attention_mask and embeddings must be on "
                "the same device."
            )

        if multimodal_cls_index is not None:
            sequence_length = embeddings.shape[1]

            if not (
                0
                <= multimodal_cls_index
                < sequence_length
            ):
                raise ValueError(
                    "multimodal_cls_index is outside the "
                    "multimodal sequence."
                )