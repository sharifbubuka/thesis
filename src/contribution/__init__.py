from __future__ import annotations

from typing import Any

__all__ = [
    "ContributionResult",
    "TargetSpec",
    "ModalityContributionEstimator",
    "estimate_samples",
    "save_contribution_artifacts",
]


def __getattr__(name: str) -> Any:
    if name in {"ContributionResult", "TargetSpec"}:
        from src.contribution.contracts import ContributionResult, TargetSpec
        return {"ContributionResult": ContributionResult, "TargetSpec": TargetSpec}[name]
    if name == "ModalityContributionEstimator":
        from src.contribution.estimator import ModalityContributionEstimator
        return ModalityContributionEstimator
    if name in {"estimate_samples", "save_contribution_artifacts"}:
        from src.contribution.pipeline import estimate_samples, save_contribution_artifacts
        return {
            "estimate_samples": estimate_samples,
            "save_contribution_artifacts": save_contribution_artifacts,
        }[name]
    raise AttributeError(name)
