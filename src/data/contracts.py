from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class CanonicalSample:
    """Canonical image-text sample shared by every benchmark pipeline."""

    sample_id: str
    benchmark: str
    images: tuple[Image.Image, ...]
    text: str
    target: str | int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("A canonical sample must have a non-empty sample ID.")
        if not self.benchmark.strip():
            raise ValueError("A canonical sample must identify its benchmark.")
        if not self.images:
            raise ValueError("A multimodal sample must contain at least one image.")
        if not self.text.strip():
            raise ValueError("A multimodal sample must contain non-empty text.")


# Compatibility name used by the existing benchmark and contribution APIs.
MultimodalSample = CanonicalSample
