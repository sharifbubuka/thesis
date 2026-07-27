from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TargetSpec:
    """The scalar model output whose contribution is being explained."""

    target_id: int | None
    target_label: str
    mode: str = "predicted"


@dataclass
class ContributionResult:
    sample_id: str
    benchmark: str
    checkpoint: str
    prediction: str
    predicted_id: int | None
    target_label: str | None
    is_correct: bool | None

    confidence: float
    margin: float
    entropy: float

    full_score: float
    image_only_score: float
    text_only_score: float
    baseline_score: float

    image_contribution: float
    text_contribution: float
    interaction_contribution: float
    total_effect: float

    image_share: float
    text_share: float
    interaction_share: float

    image_gradient_norm: float
    text_gradient_norm: float
    image_gradient_x_input: float
    text_gradient_x_input: float

    top_patch_index: int | None
    top_patch_score: float | None
    top_token: str | None
    top_token_index: int | None
    top_token_score: float | None

    cls_embedding: np.ndarray | None = field(default=None, repr=False)
    image_patch_scores: np.ndarray | None = field(default=None, repr=False)
    text_token_scores: np.ndarray | None = field(default=None, repr=False)
    tokens: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def scalar_record(self) -> dict[str, Any]:
        """Return only DataFrame-friendly scalar values."""
        return {
            "sample_id": self.sample_id,
            "benchmark": self.benchmark,
            "checkpoint": self.checkpoint,
            "prediction": self.prediction,
            "predicted_id": self.predicted_id,
            "target_label": self.target_label,
            "is_correct": self.is_correct,
            "confidence": self.confidence,
            "margin": self.margin,
            "entropy": self.entropy,
            "full_score": self.full_score,
            "image_only_score": self.image_only_score,
            "text_only_score": self.text_only_score,
            "baseline_score": self.baseline_score,
            "image_contribution": self.image_contribution,
            "text_contribution": self.text_contribution,
            "interaction_contribution": self.interaction_contribution,
            "total_effect": self.total_effect,
            "image_share": self.image_share,
            "text_share": self.text_share,
            "interaction_share": self.interaction_share,
            "image_gradient_norm": self.image_gradient_norm,
            "text_gradient_norm": self.text_gradient_norm,
            "image_gradient_x_input": self.image_gradient_x_input,
            "text_gradient_x_input": self.text_gradient_x_input,
            "top_patch_index": self.top_patch_index,
            "top_patch_score": self.top_patch_score,
            "top_token": self.top_token,
            "top_token_index": self.top_token_index,
            "top_token_score": self.top_token_score,
            "tokens": " ".join(self.tokens),
            **{f"metadata_{key}": value for key, value in self.metadata.items() if np.isscalar(value)},
        }
