from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.benchmarks.contracts import MultimodalSample
from src.contribution.contracts import ContributionResult
from src.contribution.estimator import ModalityContributionEstimator


def estimate_samples(
    estimator: ModalityContributionEstimator,
    samples: Iterable[MultimodalSample],
    *,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, list[ContributionResult]]:
    materialized = list(samples)
    iterator = tqdm(materialized, desc="Estimating modality contribution") if show_progress else materialized
    results = [estimator.estimate(sample) for sample in iterator]
    dataframe = pd.DataFrame(result.scalar_record() for result in results)
    return dataframe, results


def save_contribution_artifacts(
    dataframe: pd.DataFrame,
    results: list[ContributionResult],
    output_dir: Path,
    *,
    stem: str = "stage_2_modality_contribution",
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}.csv"
    dataframe.to_csv(csv_path, index=False)

    arrays: dict[str, np.ndarray] = {}
    for result in results:
        prefix = result.sample_id.replace("/", "_")
        if result.cls_embedding is not None:
            arrays[f"{prefix}__cls"] = result.cls_embedding
        if result.image_patch_scores is not None:
            arrays[f"{prefix}__patch_scores"] = result.image_patch_scores
        if result.text_token_scores is not None:
            arrays[f"{prefix}__token_scores"] = result.text_token_scores
    npz_path = output_dir / f"{stem}_arrays.npz"
    np.savez_compressed(npz_path, **arrays)
    return {"dataframe": csv_path, "arrays": npz_path}
