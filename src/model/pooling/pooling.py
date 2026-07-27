from __future__ import annotations

from typing import Any

from torch import Tensor, nn


class MultimodalPooler(nn.Module):
    """
    Aggregate the fused multimodal sequence into one vector.

    Currently supported strategies:
        cls:
            Select the multimodal CLS token.

        mean:
            Compute a masked mean over all valid tokens.
    """

    SUPPORTED_STRATEGIES = {
        "cls",
        "mean",
    }

    def __init__(
        self,
        config: dict[str, Any],
    ) -> None:
        super().__init__()

        pooling_config = config["pooling"]

        self.hidden_size: int = config["fusion"]["hidden_size"]
        self.strategy: str = pooling_config["strategy"]
        self.use_layer_norm: bool = pooling_config[
            "use_layer_norm"
        ]
        self.dropout_probability: float = pooling_config[
            "dropout"
        ]

        self._validate_config()

        if self.use_layer_norm:
            self.layer_norm: nn.Module = nn.LayerNorm(
                self.hidden_size
            )
        else:
            self.layer_norm = nn.Identity()

        self.dropout = nn.Dropout(
            self.dropout_probability
        )

    def _validate_config(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError(
                "Pooling hidden size must be positive."
            )

        if self.strategy not in self.SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unsupported pooling strategy '{self.strategy}'. "
                f"Expected one of "
                f"{sorted(self.SUPPORTED_STRATEGIES)}."
            )

        if not 0.0 <= self.dropout_probability < 1.0:
            raise ValueError(
                "Pooling dropout must be in the range [0, 1)."
            )

    def forward(
        self,
        *,
        hidden_states: Tensor,
        attention_mask: Tensor,
        multimodal_cls_index: int | None,
    ) -> Tensor:
        """
        Pool a fused multimodal sequence.

        Args:
            hidden_states:
                [batch_size, sequence_length, hidden_size]

            attention_mask:
                [batch_size, sequence_length]

            multimodal_cls_index:
                CLS position when CLS pooling is configured.

        Returns:
            Pooled tensor with shape:
            [batch_size, hidden_size]
        """

        self._validate_inputs(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            multimodal_cls_index=multimodal_cls_index,
        )

        if self.strategy == "cls":
            if multimodal_cls_index is None:
                raise ValueError(
                    "CLS pooling requires a multimodal CLS token."
                )

            pooled_output = hidden_states[
                :,
                multimodal_cls_index,
                :,
            ]
        else:
            mask = attention_mask.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            ).unsqueeze(-1)

            masked_hidden_states = hidden_states * mask

            valid_token_counts = mask.sum(
                dim=1
            ).clamp_min(1.0)

            pooled_output = (
                masked_hidden_states.sum(dim=1)
                / valid_token_counts
            )

        pooled_output = self.layer_norm(
            pooled_output
        )

        return self.dropout(
            pooled_output
        )

    def _validate_inputs(
        self,
        *,
        hidden_states: Tensor,
        attention_mask: Tensor,
        multimodal_cls_index: int | None,
    ) -> None:
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape "
                "[batch_size, sequence_length, hidden_size]."
            )

        if attention_mask.ndim != 2:
            raise ValueError(
                "attention_mask must have shape "
                "[batch_size, sequence_length]."
            )

        if hidden_states.shape[:2] != attention_mask.shape:
            raise ValueError(
                "attention_mask must match the batch and sequence "
                "dimensions of hidden_states."
            )

        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                "Unexpected pooling hidden size: "
                f"expected={self.hidden_size}, "
                f"received={hidden_states.shape[-1]}."
            )

        if not hidden_states.is_floating_point():
            raise TypeError(
                "hidden_states must use a floating-point dtype."
            )

        if multimodal_cls_index is not None:
            sequence_length = hidden_states.shape[1]

            if not 0 <= multimodal_cls_index < sequence_length:
                raise ValueError(
                    "multimodal_cls_index is outside the "
                    "multimodal sequence."
                )