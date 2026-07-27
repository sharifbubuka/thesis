from __future__ import annotations

from typing import Any

import torch
from transformers import (
    ViltForImageAndTextRetrieval,
    ViltForImagesAndTextClassification,
    ViltForQuestionAnswering,
    ViltProcessor,
)

from src.benchmarks.registry import BenchmarkName, CheckpointSpec, get_checkpoint_spec

ModelType = (
    ViltForQuestionAnswering
    | ViltForImagesAndTextClassification
    | ViltForImageAndTextRetrieval
)

_MODEL_CLASSES: dict[BenchmarkName, type[Any]] = {
    BenchmarkName.VQAV2: ViltForQuestionAnswering,
    BenchmarkName.NLVR2: ViltForImagesAndTextClassification,
    BenchmarkName.COCO_RETRIEVAL: ViltForImageAndTextRetrieval,
}


def load_vilt_checkpoint(
    benchmark: BenchmarkName | str,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> tuple[CheckpointSpec, ViltProcessor, ModelType]:
    spec = get_checkpoint_spec(benchmark)
    processor = ViltProcessor.from_pretrained(spec.checkpoint)
    model_class = _MODEL_CLASSES[spec.benchmark]

    kwargs: dict[str, Any] = {}
    if dtype is not None:
        kwargs["torch_dtype"] = dtype

    kwargs["attn_implementation"] = "eager"
    model = model_class.from_pretrained(spec.checkpoint, **kwargs)
    model.to(device)
    model.eval()
    return spec, processor, model
