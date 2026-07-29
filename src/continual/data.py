from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
from datasets import load_dataset  # type: ignore[import-untyped]
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import ViltProcessor

from src.continual.registry import ContinualTask, VqaDatasetSpec
from src.continual.vocabulary import AnswerVocabulary, normalize_answer
from src.data.contracts import CanonicalSample


def _majority_answer(answers: Sequence[str]) -> str:
    normalized = [normalize_answer(value) for value in answers]
    normalized = [value for value in normalized if value]
    if not normalized:
        return ""
    counts = Counter(normalized)
    return sorted(counts, key=lambda value: (-counts[value], value))[0]


def _answers_from_row(task: ContinualTask, row: Mapping[str, Any]) -> tuple[str, ...]:
    if task is ContinualTask.GQA:
        return (str(row["answer"]),)
    answers = row.get("answers", ())
    return tuple(
        str(answer.get("answer", "")) if isinstance(answer, Mapping) else str(answer)
        for answer in answers
    )


def _convert_row(
    spec: VqaDatasetSpec,
    row: Mapping[str, Any],
    *,
    split: str,
    image: Image.Image | None = None,
) -> CanonicalSample:
    task = spec.name
    answers = _answers_from_row(task, row)
    if task is ContinualTask.TEXTVQA:
        sample_id = f"textvqa-{row['question_id']}"
        image_id = str(row["image_id"])
    elif task is ContinualTask.GQA:
        sample_id = f"gqa-{row['id']}"
        image_id = str(row["imageId"])
    else:
        sample_id = f"vqav2-{row['question_id']}"
        image_id = str(row["image_id"])
    source_image = image if image is not None else row["image"]
    target = _majority_answer(answers)
    return CanonicalSample(
        sample_id=sample_id,
        benchmark=task.value,
        images=(source_image.convert("RGB"),),
        text=str(row["question"]),
        target=target,
        metadata={
            "source_dataset": spec.repository,
            "split": split,
            "image_id": image_id,
            "answers": answers,
        },
    )


def _take_shuffled(dataset: Iterable[Mapping[str, Any]], count: int, seed: int) -> list[Mapping[str, Any]]:
    shuffled = dataset.shuffle(seed=seed, buffer_size=max(1_000, count * 10))  # type: ignore[attr-defined]
    return list(shuffled.take(count))  # type: ignore[attr-defined]


def _load_gqa(
    spec: VqaDatasetSpec,
    *,
    split: str,
    config_prefix: str,
    count: int,
    seed: int,
) -> list[CanonicalSample]:
    instructions = load_dataset(
        spec.repository,
        f"{config_prefix}_instructions",
        split=split,
        streaming=True,
    )
    rows = _take_shuffled(instructions, count, seed)
    required_image_ids = {str(row["imageId"]) for row in rows}
    images = load_dataset(
        spec.repository,
        f"{config_prefix}_images",
        split=split,
        streaming=True,
    )
    image_by_id: dict[str, Image.Image] = {}
    for image_row in images:
        image_id = str(image_row["id"])
        if image_id in required_image_ids:
            image_by_id[image_id] = image_row["image"]
            if len(image_by_id) == len(required_image_ids):
                break
    missing = required_image_ids.difference(image_by_id)
    if missing:
        raise RuntimeError(f"GQA image stream did not contain {len(missing)} selected image(s).")
    return [
        _convert_row(
            spec,
            row,
            split=split,
            image=image_by_id[str(row["imageId"])],
        )
        for row in rows
    ]


def load_vqa_samples(
    spec: VqaDatasetSpec,
    *,
    training: bool,
    count: int,
    seed: int = 42,
) -> list[CanonicalSample]:
    if count <= 0:
        raise ValueError("count must be positive.")
    split = spec.train_split if training else spec.validation_split
    config = spec.train_config if training else spec.validation_config
    if spec.separate_images:
        if config is None:
            raise ValueError("A GQA-style source requires a config prefix.")
        return _load_gqa(spec, split=split, config_prefix=config, count=count, seed=seed)
    dataset = load_dataset(
        spec.repository,
        config,
        split=split,
        streaming=True,
    )
    rows = _take_shuffled(dataset, count, seed)
    return [_convert_row(spec, row, split=split) for row in rows]


class VqaSampleDataset(Dataset[CanonicalSample]):
    def __init__(self, samples: Sequence[CanonicalSample]) -> None:
        self.samples = tuple(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> CanonicalSample:
        return self.samples[index]


class VqaCollator:
    def __init__(
        self,
        processor: ViltProcessor,
        vocabulary: AnswerVocabulary,
        *,
        max_length: int,
    ) -> None:
        self.processor = processor
        self.vocabulary = vocabulary
        self.max_length = max_length

    def __call__(self, samples: Sequence[CanonicalSample]) -> dict[str, Any]:
        encoding = self.processor(
            images=[sample.images[0] for sample in samples],
            text=[sample.text for sample in samples],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = torch.zeros((len(samples), len(self.vocabulary.answers)), dtype=torch.float32)
        for row_index, sample in enumerate(samples):
            raw_answers = sample.metadata.get("answers", ())
            answers = tuple(str(answer) for answer in raw_answers) or (str(sample.target),)
            counts = Counter(self.vocabulary.encode(answer) for answer in answers)
            for answer_id, count in counts.items():
                labels[row_index, answer_id] = min(float(count) / 3.0, 1.0)
        return {"inputs": dict(encoding), "labels": labels, "samples": tuple(samples)}


def make_loader(
    samples: Sequence[CanonicalSample],
    *,
    processor: ViltProcessor,
    vocabulary: AnswerVocabulary,
    max_length: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader[CanonicalSample]:
    return DataLoader(
        VqaSampleDataset(samples),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=VqaCollator(processor, vocabulary, max_length=max_length),
    )
