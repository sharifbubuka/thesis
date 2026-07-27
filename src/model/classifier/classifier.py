from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from ..contracts import ClassificationHeadOutput

class VQAClassificationHead(nn.Module):
    """
    Predict soft VQA answer scores from a fused CLS embedding.

    The head returns one independent logit for every answer
    in the fixed answer vocabulary.
    """

    def __init__(
        self,
        config: dict[str, Any],
    ) -> None:
        super().__init__()

        classification_config = config["classifier"]

        self.input_size: int = config["fusion"]["hidden_size"]
        self.hidden_size: int = classification_config["hidden_size"]
        self.number_of_classes: int = classification_config[
            "number_of_classes"
        ]
        self.dropout_probability: float = classification_config[
            "dropout"
        ]
        self.activation_name: str = classification_config[
            "activation"
        ]
        self.use_layer_norm: bool = classification_config[
            "use_layer_norm"
        ]
        self.use_bias: bool = classification_config["use_bias"]

        self._validate_config()

        self.layer_norm: nn.Module

        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(self.input_size)
        else:
            self.layer_norm = nn.Identity()

        self.hidden_projection = nn.Linear(
            self.input_size,
            self.hidden_size,
            bias=self.use_bias,
        )

        self.activation = self._build_activation()

        self.dropout = nn.Dropout(
            self.dropout_probability
        )

        self.output_projection = nn.Linear(
            self.hidden_size,
            self.number_of_classes,
            bias=self.use_bias,
        )

        self.reset_parameters()

    def _validate_config(self) -> None:
        if self.input_size <= 0:
            raise ValueError(
                "Classification input size must be positive."
            )

        if self.hidden_size <= 0:
            raise ValueError(
                "Classification hidden size must be positive."
            )

        if self.number_of_classes <= 1:
            raise ValueError(
                "number_of_classes must be greater than one."
            )

        if not 0.0 <= self.dropout_probability < 1.0:
            raise ValueError(
                "Classification dropout must be in [0, 1)."
            )

        if self.activation_name not in {
            "gelu",
            "relu",
            "silu",
        }:
            raise ValueError(
                "Unsupported classification activation: "
                f"{self.activation_name}."
            )

    def _build_activation(self) -> nn.Module:
        activations: dict[str, nn.Module] = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
        }

        return activations[self.activation_name]

    def reset_parameters(self) -> None:
        nn.init.normal_(
            self.hidden_projection.weight,
            mean=0.0,
            std=0.02,
        )

        nn.init.normal_(
            self.output_projection.weight,
            mean=0.0,
            std=0.02,
        )

        if self.hidden_projection.bias is not None:
            nn.init.zeros_(self.hidden_projection.bias)

        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        cls_embedding: Tensor,
    ) -> ClassificationHeadOutput:
        self._validate_input(cls_embedding)

        hidden_states = self.layer_norm(
            cls_embedding
        )

        hidden_states = self.hidden_projection(
            hidden_states
        )

        hidden_states = self.activation(
            hidden_states
        )

        hidden_states = self.dropout(
            hidden_states
        )

        logits = self.output_projection(
            hidden_states
        )

        scores = torch.sigmoid(logits)

        confidence, predicted_ids = scores.max(
            dim=-1
        )

        epsilon = torch.finfo(scores.dtype).eps

        stable_scores = scores.clamp(
            min=epsilon,
            max=1.0 - epsilon,
        )

        binary_entropy = -(
            stable_scores * stable_scores.log()
            + (1.0 - stable_scores)
            * (1.0 - stable_scores).log()
        ).mean(dim=-1)

        return ClassificationHeadOutput(
            logits=logits,
            scores=scores,
            predicted_ids=predicted_ids,
            confidence=confidence,
            binary_entropy=binary_entropy,
        )

    def _validate_input(
        self,
        cls_embedding: Tensor,
    ) -> None:
        if cls_embedding.ndim != 2:
            raise ValueError(
                "cls_embedding must have shape "
                "[batch_size, hidden_size]."
            )

        if cls_embedding.shape[1] != self.input_size:
            raise ValueError(
                "Unexpected CLS hidden size: "
                f"expected={self.input_size}, "
                f"received={cls_embedding.shape[1]}."
            )

        if not cls_embedding.is_floating_point():
            raise TypeError(
                "cls_embedding must use a floating-point dtype."
            )