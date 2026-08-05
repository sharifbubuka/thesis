from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from transformers import ViltProcessor

from src.benchmarks.contracts import MultimodalSample
from src.benchmarks.registry import BenchmarkName, CheckpointSpec
from src.vilt.loader import ModelType
from src.vilt.outputs import PredictionResult


def _detach_to_cpu(values: Any) -> Any:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu()
    if isinstance(values, (tuple, list)):
        return tuple(_detach_to_cpu(value) for value in values)
    return values


class ViltAdapter(ABC):
    def __init__(
        self,
        spec: CheckpointSpec,
        processor: ViltProcessor,
        model: ModelType,
        device: torch.device,
    ) -> None:
        self.spec = spec
        self.processor = processor
        self.model = model
        self.device = device

    def _validate_sample(self, sample: MultimodalSample) -> None:
        if len(sample.images) != self.spec.expected_images:
            raise ValueError(
                f"{self.spec.benchmark.value} expects {self.spec.expected_images} image(s), "
                f"but received {len(sample.images)}."
            )

    @abstractmethod
    def _prepare_inputs(self, sample: MultimodalSample) -> dict[str, torch.Tensor]: ...

    def prepare_inputs(self, sample: MultimodalSample) -> dict[str, torch.Tensor]:
        self._validate_sample(sample)
        return self._prepare_inputs(sample)

    @abstractmethod
    def probabilities(self, logits: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def select_target_score(
        self, logits: torch.Tensor, target_id: int | None
    ) -> torch.Tensor: ...

    @abstractmethod
    def _decode(
        self, logits: torch.Tensor, sample: MultimodalSample
    ) -> tuple[str, int | None, torch.Tensor]: ...

    def predict(
        self,
        sample: MultimodalSample,
        *,
        include_internals: bool = True,
        gradients_enabled: bool = False,
    ) -> PredictionResult:
        self._validate_sample(sample)
        inputs = self.prepare_inputs(sample)
        context = torch.enable_grad() if gradients_enabled else torch.inference_mode()

        with context:
            outputs = self.model(
                **inputs,
                output_hidden_states=include_internals,
                output_attentions=include_internals,
                return_dict=True,
            )

        logits = outputs.logits
        if not torch.isfinite(logits).all():
            raise RuntimeError("The model produced non-finite logits.")

        predicted_label, predicted_id, scores = self._decode(logits, sample)
        target_label = None if sample.target is None else str(sample.target)
        is_correct = None if target_label is None else predicted_label.lower() == target_label.lower()
        tokens = tuple(self.processor.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))

        return PredictionResult(
            sample_id=sample.sample_id,
            benchmark=self.spec.benchmark.value,
            checkpoint=self.spec.checkpoint,
            predicted_label=predicted_label,
            predicted_id=predicted_id,
            target_label=target_label,
            is_correct=is_correct,
            logits=logits.detach().cpu(),
            scores=scores.detach().cpu(),
            input_shapes={key: tuple(value.shape) for key, value in inputs.items()},
            hidden_states=(
                _detach_to_cpu(outputs.hidden_states)
                if include_internals and outputs.hidden_states is not None
                else None
            ),
            attentions=(
                _detach_to_cpu(outputs.attentions)
                if include_internals and outputs.attentions is not None
                else None
            ),
            tokens=tokens,
            metadata=dict(sample.metadata),
        )


class VqaAdapter(ViltAdapter):
    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits)

    def select_target_score(self, logits: torch.Tensor, target_id: int | None) -> torch.Tensor:
        if target_id is None:
            raise ValueError("VQA attribution requires an answer target ID.")
        return logits[0, target_id]

    def _prepare_inputs(self, sample: MultimodalSample) -> dict[str, torch.Tensor]:
        encoding = self.processor(
            images=sample.images[0],
            text=sample.text,
            truncation=True,
            max_length=self.model.config.max_position_embeddings,
            return_tensors="pt",
        )
        return {key: value.to(self.device) for key, value in encoding.items()}

    def _decode(
        self, logits: torch.Tensor, sample: MultimodalSample
    ) -> tuple[str, int, torch.Tensor]:
        probabilities = torch.sigmoid(logits)
        predicted_id = int(logits.argmax(dim=-1).item())
        predicted_label = str(self.model.config.id2label[predicted_id])
        return predicted_label, predicted_id, probabilities


class Nlvr2Adapter(ViltAdapter):
    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=-1)

    def select_target_score(self, logits: torch.Tensor, target_id: int | None) -> torch.Tensor:
        if target_id is None:
            raise ValueError("NLVR2 attribution requires a class target ID.")
        return logits[0, target_id]

    def _prepare_inputs(self, sample: MultimodalSample) -> dict[str, torch.Tensor]:
        encoding = self.processor(
            images=list(sample.images),
            text=sample.text,
            truncation=True,
            max_length=self.model.config.max_position_embeddings,
            return_tensors="pt",
        )
        # ViLT's NLVR2 head expects [batch, num_images, channels, height, width].
        pixel_values = encoding["pixel_values"].unsqueeze(0)
        inputs: dict[str, torch.Tensor] = {
            "input_ids": encoding["input_ids"].to(self.device),
            "pixel_values": pixel_values.to(self.device),
        }
        if "attention_mask" in encoding:
            inputs["attention_mask"] = encoding["attention_mask"].to(self.device)
        if "token_type_ids" in encoding:
            inputs["token_type_ids"] = encoding["token_type_ids"].to(self.device)
        return inputs

    def _decode(
        self, logits: torch.Tensor, sample: MultimodalSample
    ) -> tuple[str, int, torch.Tensor]:
        probabilities = torch.softmax(logits, dim=-1)
        predicted_id = int(logits.argmax(dim=-1).item())
        predicted_label = str(self.model.config.id2label[predicted_id])
        return predicted_label, predicted_id, probabilities


class CocoRetrievalAdapter(ViltAdapter):
    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits.reshape(-1))

    def select_target_score(self, logits: torch.Tensor, target_id: int | None) -> torch.Tensor:
        del target_id
        return logits.reshape(-1)[0]

    def _prepare_inputs(self, sample: MultimodalSample) -> dict[str, torch.Tensor]:
        encoding = self.processor(
            images=sample.images[0],
            text=sample.text,
            truncation=True,
            max_length=self.model.config.max_position_embeddings,
            return_tensors="pt",
        )
        return {key: value.to(self.device) for key, value in encoding.items()}

    def _decode(
        self, logits: torch.Tensor, sample: MultimodalSample
    ) -> tuple[str, None, torch.Tensor]:
        del sample
        confidence = torch.sigmoid(logits.reshape(-1))
        percentage = float(confidence[0].item() * 100.0)
        return f"match_confidence={percentage:.2f}%", None, confidence


def create_adapter(
    spec: CheckpointSpec,
    processor: ViltProcessor,
    model: ModelType,
    device: torch.device,
) -> ViltAdapter:
    if spec.benchmark is BenchmarkName.VQAV2:
        return VqaAdapter(spec, processor, model, device)
    if spec.benchmark is BenchmarkName.NLVR2:
        return Nlvr2Adapter(spec, processor, model, device)
    if spec.benchmark is BenchmarkName.COCO_RETRIEVAL:
        return CocoRetrievalAdapter(spec, processor, model, device)
    raise AssertionError(f"Unhandled benchmark: {spec.benchmark}")
