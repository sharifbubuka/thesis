from PIL import Image

from src.benchmarks.contracts import MultimodalSample
from src.contribution.ablation import (
    ablated_sample,
    inclusion_exclusion,
    normalized_absolute_shares,
)


def test_inclusion_exclusion_reconstructs_total_effect() -> None:
    image, text, interaction, total = inclusion_exclusion(
        full=10.0, image_only=5.0, text_only=4.0, baseline=1.0
    )
    assert image == 4.0
    assert text == 3.0
    assert interaction == 2.0
    assert total == 9.0
    assert image + text + interaction == total


def test_absolute_shares_sum_to_one() -> None:
    shares = normalized_absolute_shares(-2.0, 1.0, 1.0)
    assert abs(sum(shares) - 1.0) < 1e-12


def test_ablation_preserves_image_count_and_valid_text() -> None:
    sample = MultimodalSample(
        sample_id="sample",
        benchmark="vqav2",
        images=(Image.new("RGB", (10, 10), "white"),),
        text="What is shown?",
    )
    ablated = ablated_sample(sample, remove_images=True, remove_text=True)
    assert len(ablated.images) == 1
    assert ablated.images[0].size == sample.images[0].size
    assert ablated.text == "[MASK]"
