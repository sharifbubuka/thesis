from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class PredictionResult:
    sample_id: str
    benchmark: str
    checkpoint: str
    predicted_label: str
    predicted_id: int | None
    target_label: str | None
    is_correct: bool | None
    logits: torch.Tensor
    scores: torch.Tensor
    input_shapes: dict[str, tuple[int, ...]]
    hidden_states: tuple[torch.Tensor, ...] | tuple[tuple[torch.Tensor, ...], ...] | None
    attentions: tuple[torch.Tensor, ...] | tuple[tuple[torch.Tensor, ...], ...] | None
    tokens: tuple[str, ...]
    metadata: dict[str, Any]

    @staticmethod
    def _shape_tree(values: object) -> object:
        if isinstance(values, torch.Tensor):
            return list(values.shape)
        if isinstance(values, (tuple, list)):
            return [PredictionResult._shape_tree(value) for value in values]
        return None

    @property
    def hidden_state_shapes(self) -> object:
        return self._shape_tree(self.hidden_states)

    @property
    def attention_shapes(self) -> object:
        return self._shape_tree(self.attentions)

    def summary(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "benchmark": self.benchmark,
            "checkpoint": self.checkpoint,
            "predicted_label": self.predicted_label,
            "predicted_id": self.predicted_id,
            "target_label": self.target_label,
            "is_correct": self.is_correct,
            "logits_shape": list(self.logits.shape),
            "input_shapes": {key: list(value) for key, value in self.input_shapes.items()},
            "hidden_state_groups": len(self.hidden_states or ()),
            "hidden_state_shapes": self.hidden_state_shapes,
            "attention_groups": len(self.attentions or ()),
            "attention_shapes": self.attention_shapes,
            "tokens": list(self.tokens),
            "forward_pass_valid": bool(torch.isfinite(self.logits).all().item()),
            "metadata": self.metadata,
        }
