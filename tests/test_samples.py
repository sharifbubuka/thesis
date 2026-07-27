from PIL import Image
import pytest

from src.benchmarks.contracts import MultimodalSample


def test_sample_requires_an_image() -> None:
    with pytest.raises(ValueError, match="at least one image"):
        MultimodalSample(sample_id="x", benchmark="vqav2", images=(), text="question")


def test_sample_requires_text() -> None:
    image = Image.new("RGB", (32, 32))
    with pytest.raises(ValueError, match="non-empty text"):
        MultimodalSample(sample_id="x", benchmark="vqav2", images=(image,), text=" ")
