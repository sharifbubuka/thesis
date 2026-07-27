from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from ..contracts import ProjectionOutput


class ModalityProjection(nn.Module):
    """
    Project modality-specific embeddings into the shared fusion space.

    The same component can be used for image and text embeddings,
    although each modality has its own learned parameters.
    """

    def __init__(
        self,
        *,
        input_size: int,
        config: dict[str, Any],
        modality_name: str,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.modality_name = modality_name

        projection_config = config["projection"]

        self.output_size: int = projection_config["hidden_size"]
        self.dropout_probability: float = projection_config["dropout"]
        self.activation_name: str = projection_config["activation"]
        self.use_layer_norm: bool = projection_config["use_layer_norm"]
        self.use_bias: bool = projection_config["use_bias"]

        self._validate_config()

        self.linear = nn.Linear(
            in_features=self.input_size,
            out_features=self.output_size,
            bias=self.use_bias,
        )

        self.activation = self._build_activation()

        self.dropout = nn.Dropout(
            p=self.dropout_probability
        )

        if self.use_layer_norm:
            self.layer_norm: nn.Module = nn.LayerNorm(
                normalized_shape=self.output_size
            )
        else:
            self.layer_norm = nn.Identity()

    def _validate_config(self) -> None:
        if self.input_size <= 0:
            raise ValueError(
                "Projection input size must be positive."
            )

        if self.output_size <= 0:
            raise ValueError(
                "Projection output size must be positive."
            )

        if not 0.0 <= self.dropout_probability < 1.0:
            raise ValueError(
                "Projection dropout must be in the range [0, 1)."
            )

        supported_activations = {
            "gelu",
            "relu",
            "silu",
        }

        if self.activation_name not in supported_activations:
            raise ValueError(
                f"Unsupported projection activation "
                f"'{self.activation_name}'. "
                f"Expected one of {sorted(supported_activations)}."
            )

        if not self.modality_name:
            raise ValueError(
                "modality_name cannot be empty."
            )

    def _build_activation(self) -> nn.Module:
        activations: dict[str, nn.Module] = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
        }

        return activations[self.activation_name]

    def forward(
        self,
        embeddings: Tensor,
        attention_mask: Tensor,
    ) -> ProjectionOutput:
        """
        Project one modality into the shared hidden space.

        Args:
            embeddings:
                Tensor with shape:
                [batch_size, sequence_length, input_size]

            attention_mask:
                Tensor with shape:
                [batch_size, sequence_length]
        """

        self._validate_inputs(
            embeddings=embeddings,
            attention_mask=attention_mask,
        )

        projected = self.linear(embeddings)
        projected = self.activation(projected)
        projected = self.dropout(projected)
        projected = self.layer_norm(projected)

        return ProjectionOutput(
            embeddings=projected,
            attention_mask=attention_mask,
        )

    def _validate_inputs(
        self,
        *,
        embeddings: Tensor,
        attention_mask: Tensor,
    ) -> None:
        if embeddings.ndim != 3:
            raise ValueError(
                "embeddings must have shape "
                "[batch_size, sequence_length, hidden_size], "
                f"but received {tuple(embeddings.shape)}."
            )

        if embeddings.shape[-1] != self.input_size:
            raise ValueError(
                "Embedding hidden size does not match the "
                f"projection input size: expected={self.input_size}, "
                f"received={embeddings.shape[-1]}."
            )

        if attention_mask.ndim != 2:
            raise ValueError(
                "attention_mask must have shape "
                "[batch_size, sequence_length]."
            )

        if attention_mask.shape != embeddings.shape[:2]:
            raise ValueError(
                "attention_mask must match the batch and sequence "
                "dimensions of embeddings."
            )

        if not embeddings.is_floating_point():
            raise TypeError(
                "embeddings must use a floating-point dtype."
            )

        if attention_mask.device != embeddings.device:
            raise ValueError(
                "embeddings and attention_mask must be on the "
                "same device."
            )