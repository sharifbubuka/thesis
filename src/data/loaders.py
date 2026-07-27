from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from hashlib import sha1
from typing import Any

from datasets import load_dataset  # type: ignore[import-untyped]
from PIL import Image

from src.data.contracts import CanonicalSample


@dataclass(frozen=True)
class DatasetSource:
    repository: str
    split: str
    converter: Callable[[Mapping[str, Any]], CanonicalSample]


def _rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


def _vqa_sample(row: Mapping[str, Any]) -> CanonicalSample:
    answers = tuple(answer["answer"] for answer in row["answers"])
    return CanonicalSample(
        sample_id=f"vqav2-{row['question_id']}",
        benchmark="vqav2",
        images=(_rgb(row["image"]),),
        text=str(row["question"]),
        target=str(row["multiple_choice_answer"]),
        metadata={
            "source_dataset": "lmms-lab/VQAv2",
            "split": "validation",
            "image_id": int(row["image_id"]),
            "question_type": str(row["question_type"]),
            "answer_type": str(row["answer_type"]),
            "answers": answers,
        },
    )


def _nlvr2_sample(row: Mapping[str, Any]) -> CanonicalSample:
    return CanonicalSample(
        sample_id=f"nlvr2-{row['identifier']}",
        benchmark="nlvr2",
        images=(_rgb(row["image0"]), _rgb(row["image1"])),
        text=str(row["sentence"]),
        target=str(row["label"]),
        metadata={
            "source_dataset": "pingzhili/nlvr2",
            "split": "validation",
            "identifier": str(row["identifier"]),
        },
    )


def _coco_sample(row: Mapping[str, Any]) -> CanonicalSample:
    caption = str(row["caption"])
    caption_id = sha1(caption.encode("utf-8")).hexdigest()[:12]
    return CanonicalSample(
        sample_id=f"coco-retrieval-{row['cocoid']}-{caption_id}",
        benchmark="coco_retrieval",
        images=(_rgb(row["image"]),),
        text=caption,
        metadata={
            "source_dataset": "jxie/coco_captions",
            "split": "validation",
            "image_id": int(row["cocoid"]),
            "filename": str(row["filename"]),
        },
    )


DATASET_SOURCES: dict[str, DatasetSource] = {
    "vqav2": DatasetSource("lmms-lab/VQAv2", "validation", _vqa_sample),
    "nlvr2": DatasetSource("pingzhili/nlvr2", "validation", _nlvr2_sample),
    "coco_retrieval": DatasetSource("jxie/coco_captions", "validation", _coco_sample),
}


def _benchmark_name(benchmark: str | Enum) -> str:
    value = getattr(benchmark, "value", benchmark)
    if not isinstance(value, str):
        raise TypeError("benchmark must be a string or string-valued enum.")
    return value


def load_benchmark_samples(
    benchmark: str | Enum,
    *,
    count: int,
    seed: int = 42,
    shuffle_buffer_size: int = 100,
) -> list[CanonicalSample]:
    """Load a deterministic sample from a benchmark's validation split."""
    if count <= 0:
        raise ValueError("count must be positive.")

    name = _benchmark_name(benchmark)
    try:
        source = DATASET_SOURCES[name]
    except KeyError as error:
        supported = ", ".join(sorted(DATASET_SOURCES))
        raise ValueError(f"Unsupported benchmark {name!r}. Choose from: {supported}.") from error

    dataset = load_dataset(source.repository, split=source.split, streaming=True)
    rows: Iterable[Mapping[str, Any]] = dataset.shuffle(
        seed=seed,
        buffer_size=max(count, shuffle_buffer_size),
    ).take(count)
    samples = [source.converter(row) for row in rows]
    if len(samples) != count:
        raise RuntimeError(
            f"{source.repository}:{source.split} returned {len(samples)} rows; expected {count}."
        )
    return samples


def load_balanced_benchmark_samples(
    *,
    count_per_benchmark: int,
    seed: int = 42,
) -> dict[str, list[CanonicalSample]]:
    """Load the same number of canonical samples for every configured benchmark."""
    return {
        benchmark: load_benchmark_samples(benchmark, count=count_per_benchmark, seed=seed)
        for benchmark in DATASET_SOURCES
    }
