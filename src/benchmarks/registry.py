from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class BenchmarkName(StrEnum):
    VQAV2 = "vqav2"
    NLVR2 = "nlvr2"
    COCO_RETRIEVAL = "coco_retrieval"


class TaskKind(StrEnum):
    QUESTION_ANSWERING = "question_answering"
    IMAGE_PAIR_CLASSIFICATION = "image_pair_classification"
    IMAGE_TEXT_RETRIEVAL = "image_text_retrieval"


@dataclass(frozen=True)
class CheckpointSpec:
    benchmark: BenchmarkName
    checkpoint: str
    model_class_name: str
    task: TaskKind
    expected_images: int


CHECKPOINTS: dict[BenchmarkName, CheckpointSpec] = {
    BenchmarkName.VQAV2: CheckpointSpec(
        benchmark=BenchmarkName.VQAV2,
        checkpoint="dandelin/vilt-b32-finetuned-vqa",
        model_class_name="ViltForQuestionAnswering",
        task=TaskKind.QUESTION_ANSWERING,
        expected_images=1,
    ),
    BenchmarkName.NLVR2: CheckpointSpec(
        benchmark=BenchmarkName.NLVR2,
        checkpoint="dandelin/vilt-b32-finetuned-nlvr2",
        model_class_name="ViltForImagesAndTextClassification",
        task=TaskKind.IMAGE_PAIR_CLASSIFICATION,
        expected_images=2,
    ),
    BenchmarkName.COCO_RETRIEVAL: CheckpointSpec(
        benchmark=BenchmarkName.COCO_RETRIEVAL,
        checkpoint="dandelin/vilt-b32-finetuned-coco",
        model_class_name="ViltForImageAndTextRetrieval",
        task=TaskKind.IMAGE_TEXT_RETRIEVAL,
        expected_images=1,
    ),
}


def get_checkpoint_spec(name: BenchmarkName | str) -> CheckpointSpec:
    try:
        benchmark = BenchmarkName(name)
    except ValueError as exc:
        supported = ", ".join(item.value for item in BenchmarkName)
        raise ValueError(f"Unknown benchmark {name!r}. Supported: {supported}") from exc
    return CHECKPOINTS[benchmark]
