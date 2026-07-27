from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.contribution.contracts import ContributionResult


def plot_contribution_distribution(
    dataframe: pd.DataFrame,
    *,
    column: str = "image_share",
    bins: int = 20,
    output_path: Path | None = None,
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.hist(dataframe[column].dropna(), bins=bins)
    axis.set_title(f"Distribution of {column.replace('_', ' ')}")
    axis.set_xlabel(column.replace("_", " ").title())
    axis.set_ylabel("Samples")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    _save(figure, output_path)
    return figure


def plot_image_vs_text(
    dataframe: pd.DataFrame,
    *,
    output_path: Path | None = None,
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(6.5, 6))
    axis.scatter(dataframe["image_share"], dataframe["text_share"], alpha=0.7)
    axis.set_title("Per-sample image and text contribution")
    axis.set_xlabel("Image contribution share")
    axis.set_ylabel("Text contribution share")
    axis.set_xlim(-0.02, 1.02)
    axis.set_ylim(-0.02, 1.02)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    _save(figure, output_path)
    return figure


def plot_benchmark_means(
    dataframe: pd.DataFrame,
    *,
    output_path: Path | None = None,
) -> plt.Figure:
    columns = ["image_share", "text_share", "interaction_share"]
    means = dataframe.groupby("benchmark")[columns].mean()
    figure, axis = plt.subplots(figsize=(9, 5))
    means.plot(kind="bar", ax=axis)
    axis.set_title("Mean modality contribution by benchmark")
    axis.set_xlabel("Benchmark")
    axis.set_ylabel("Mean absolute contribution share")
    axis.set_ylim(0, 1)
    axis.tick_params(axis="x", rotation=0)
    axis.legend(title="Component")
    figure.tight_layout()
    _save(figure, output_path)
    return figure


def plot_token_importance(
    result: ContributionResult,
    *,
    top_k: int = 20,
    output_path: Path | None = None,
) -> plt.Figure:
    if result.text_token_scores is None:
        raise ValueError("The result has no text token scores.")
    count = min(top_k, len(result.text_token_scores), len(result.tokens))
    scores = np.asarray(result.text_token_scores[:count])
    tokens = list(result.tokens[:count])

    figure, axis = plt.subplots(figsize=(max(8, count * 0.45), 4.5))
    axis.bar(tokens, scores)
    axis.set_title(f"Token importance: {result.sample_id}")
    axis.set_xlabel("Token")
    axis.set_ylabel("|gradient × activation|")
    axis.tick_params(axis="x", rotation=60)
    figure.tight_layout()
    _save(figure, output_path)
    return figure


def plot_patch_importance(
    result: ContributionResult,
    *,
    image: Image.Image | None = None,
    output_path: Path | None = None,
) -> plt.Figure:
    if result.image_patch_scores is None:
        raise ValueError("The result has no image patch scores.")
    scores = np.asarray(result.image_patch_scores)
    side = int(math.sqrt(len(scores)))

    if side * side == len(scores):
        heatmap = scores.reshape(side, side)
        figure, axis = plt.subplots(figsize=(6, 5))
        rendered = axis.imshow(heatmap)
        axis.set_title(f"Patch importance: {result.sample_id}")
        axis.set_xlabel("Patch column")
        axis.set_ylabel("Patch row")
        figure.colorbar(rendered, ax=axis, label="|gradient × activation|")
    else:
        figure, axis = plt.subplots(figsize=(9, 4.5))
        axis.bar(np.arange(len(scores)), scores)
        axis.set_title(f"Patch importance (model order): {result.sample_id}")
        axis.set_xlabel("Patch index")
        axis.set_ylabel("|gradient × activation|")

    # The optional image is deliberately not overlaid unless exact model patch
    # coordinates are available. This avoids creating a misleading spatial map.
    del image
    figure.tight_layout()
    _save(figure, output_path)
    return figure


def plot_correlation_matrix(
    dataframe: pd.DataFrame,
    *,
    output_path: Path | None = None,
) -> plt.Figure:
    preferred = [
        "confidence",
        "margin",
        "entropy",
        "image_share",
        "text_share",
        "interaction_share",
        "image_gradient_norm",
        "text_gradient_norm",
    ]
    columns = [column for column in preferred if column in dataframe]
    correlation = dataframe[columns].corr(method="spearman")

    figure, axis = plt.subplots(figsize=(8, 7))
    rendered = axis.imshow(correlation, vmin=-1, vmax=1)
    axis.set_xticks(range(len(columns)), columns, rotation=60, ha="right")
    axis.set_yticks(range(len(columns)), columns)
    axis.set_title("Spearman correlation matrix")
    figure.colorbar(rendered, ax=axis, label="Correlation")
    figure.tight_layout()
    _save(figure, output_path)
    return figure


def _save(figure: plt.Figure, output_path: Path | None) -> None:
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
