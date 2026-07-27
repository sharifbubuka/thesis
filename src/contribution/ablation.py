from __future__ import annotations

from dataclasses import replace

from PIL import Image

from src.benchmarks.contracts import MultimodalSample


def black_image_like(image: Image.Image) -> Image.Image:
    return Image.new("RGB", image.size, color=(0, 0, 0))


def ablated_sample(
    sample: MultimodalSample,
    *,
    remove_images: bool = False,
    remove_text: bool = False,
) -> MultimodalSample:
    """Create a coarse modality baseline while preserving image count and dimensions."""
    images = (
        tuple(black_image_like(image) for image in sample.images)
        if remove_images
        else sample.images
    )
    # A single tokenizer mask token keeps the sample valid and provides a
    # fixed low-information textual baseline.
    text = "[MASK]" if remove_text else sample.text
    suffix = f"-i{int(remove_images)}-t{int(remove_text)}"
    return replace(sample, sample_id=f"{sample.sample_id}{suffix}", images=images, text=text)


def inclusion_exclusion(
    *,
    full: float,
    image_only: float,
    text_only: float,
    baseline: float,
) -> tuple[float, float, float, float]:
    """Two-factor decomposition relative to a black-image/empty-text baseline.

    image contribution = f(I,T0) - f(I0,T0)
    text contribution = f(I0,T) - f(I0,T0)
    interaction = f(I,T) - f(I,T0) - f(I0,T) + f(I0,T0)
    """
    image = image_only - baseline
    text = text_only - baseline
    interaction = full - image_only - text_only + baseline
    total = full - baseline
    return image, text, interaction, total


def normalized_absolute_shares(
    image: float, text: float, interaction: float, *, epsilon: float = 1e-12
) -> tuple[float, float, float]:
    denominator = abs(image) + abs(text) + abs(interaction)
    if denominator <= epsilon:
        return 0.0, 0.0, 0.0
    return abs(image) / denominator, abs(text) / denominator, abs(interaction) / denominator
