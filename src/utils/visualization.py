from __future__ import annotations

import math
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure
from PIL import Image


def patch_scores_to_heatmap(
    patch_scores: np.ndarray | torch.Tensor,
    output_height: int,
    output_width: int,
    grid_height: int | None = None,
    grid_width: int | None = None,
) -> np.ndarray:
    """
    Convert one importance score per ViLT image patch into a heatmap.

    Args:
        patch_scores:
            Flat patch-importance array with shape [num_patches].
        output_height:
            Height of the final image-space heatmap.
        output_width:
            Width of the final image-space heatmap.
        grid_height:
            Number of patch rows. Inferred from the output aspect ratio when omitted.
        grid_width:
            Number of patch columns. Inferred from the output aspect ratio when omitted.

    Returns:
        Heatmap with shape [output_height, output_width], normalized to [0, 1].
    """
    scores = torch.as_tensor(
        patch_scores,
        dtype=torch.float32,
    ).flatten()

    num_patches = scores.numel()
    if num_patches == 0:
        raise ValueError("patch_scores must contain at least one score.")

    if (grid_height is None) != (grid_width is None):
        raise ValueError("Pass both grid_height and grid_width, or omit both.")

    if grid_height is None:
        output_aspect_ratio = output_width / output_height
        factor_pairs = [
            (height, num_patches // height)
            for height in range(1, math.isqrt(num_patches) + 1)
            if num_patches % height == 0
        ]
        orientations = factor_pairs + [(width, height) for height, width in factor_pairs]
        grid_height, grid_width = min(
            orientations,
            key=lambda shape: abs((shape[1] / shape[0]) - output_aspect_ratio),
        )

    assert grid_width is not None
    if grid_height * grid_width != num_patches:
        raise ValueError(
            f"Grid {grid_height}x{grid_width} does not match "
            f"{num_patches} patch scores."
        )

    score_grid = scores.reshape(1, 1, grid_height, grid_width)

    heatmap = F.interpolate(
        score_grid,
        size=(output_height, output_width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    heatmap -= heatmap.min()
    maximum = heatmap.max()

    if maximum > 0:
        heatmap /= maximum

    return heatmap.cpu().numpy()

def plot_patch_heatmap(
    image: Image.Image,
    patch_scores: np.ndarray | torch.Tensor,
    *,
    grid_height: int | None = None,
    grid_width: int | None = None,
    alpha: float = 0.45,
    title: str = "Image-patch contribution",
    question: str | None = None,
    predicted_answer: str | None = None,
    output_path: Path | None = None,
) -> Figure:
    """Overlay normalized patch-contribution scores on an image."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1.")

    image_rgb = image.convert("RGB")
    image_array = np.asarray(image_rgb)

    heatmap = patch_scores_to_heatmap(
        patch_scores=patch_scores,
        output_height=image_array.shape[0],
        output_width=image_array.shape[1],
        grid_height=grid_height,
        grid_width=grid_width,
    )

    figure, axis = plt.subplots(figsize=(9, 7))
    axis.imshow(image_array)
    rendered = axis.imshow(
        heatmap,
        alpha=alpha,
        cmap="jet",
        interpolation="bilinear",
    )
    figure.colorbar(rendered, ax=axis, label="Relative patch contribution")
    axis.axis("off")

    annotation_lines: list[str] = []
    if question is not None and question.strip():
        annotation_lines.append(f"Question: {textwrap.fill(question.strip(), width=80)}")
    if predicted_answer is not None and predicted_answer.strip():
        annotation_lines.append(
            f"Predicted answer: {textwrap.fill(predicted_answer.strip(), width=80)}"
        )

    if annotation_lines:
        annotation = "\n".join(annotation_lines)
        axis.set_title(title, pad=18 + 15 * len(annotation.splitlines()))
        axis.text(
            0.5,
            1.01,
            annotation,
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            wrap=True,
        )
    else:
        axis.set_title(title)
    figure.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=180, bbox_inches="tight")

    return figure
