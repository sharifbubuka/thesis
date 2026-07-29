from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ContinualTask(StrEnum):
    TEXTVQA = "textvqa"
    GQA = "gqa"
    VQAV2 = "vqav2"


@dataclass(frozen=True, slots=True)
class VqaDatasetSpec:
    name: ContinualTask
    repository: str
    train_split: str
    validation_split: str
    train_config: str | None = None
    validation_config: str | None = None
    separate_images: bool = False


CONTINUAL_TASKS: tuple[VqaDatasetSpec, ...] = (
    VqaDatasetSpec(
        name=ContinualTask.TEXTVQA,
        repository="lmms-lab/textvqa",
        train_split="train",
        validation_split="validation",
    ),
    VqaDatasetSpec(
        name=ContinualTask.GQA,
        repository="lmms-lab/GQA",
        train_split="train",
        validation_split="val",
        train_config="train_balanced",
        validation_config="val_balanced",
        separate_images=True,
    ),
    VqaDatasetSpec(
        name=ContinualTask.VQAV2,
        repository="lmms-lab/VQAv2",
        train_split="train",
        validation_split="validation",
    ),
)


def get_continual_task(name: ContinualTask | str) -> VqaDatasetSpec:
    task = ContinualTask(name)
    return next(spec for spec in CONTINUAL_TASKS if spec.name is task)
