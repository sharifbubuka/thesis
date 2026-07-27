from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.data import DataPipelineConfig, VQABatch, VQADataPipeline, VQADataPipelineOutput
from src.model import CompactViltModel
from src.model.config import CompactViltConfig
from src.model.contracts import CompactViltOutput
from src.training import TrainingConfig, TrainingHistory, VQATrainer
from .prediction import RankedAnswerPrediction, VQAPrediction
from src.utils import resolve_device, set_seed


@dataclass(frozen=True, slots=True)
class ModalityContributionPipelineConfig:
    data: DataPipelineConfig
    model: CompactViltConfig
    training: TrainingConfig | None = None
    device: str = "auto"
    return_tokens: bool = True


@dataclass(frozen=True, slots=True)
class ModalityContributionPipelineResult:
    data: VQADataPipelineOutput
    batch: VQABatch
    output: CompactViltOutput
    model: CompactViltModel
    parameter_counts: dict[str, int]
    device: torch.device
    training_history: TrainingHistory | None

    def predict(
        self,
        *,
        sample_index: int = 0,
        top_k: int = 10,
    ) -> VQAPrediction:
        batch_size = len(self.batch.samples)
        if not 0 <= sample_index < batch_size:
            raise IndexError(
                f"sample_index must be between 0 and {batch_size - 1}."
            )
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        sample = self.batch.samples[sample_index]
        scores = self.output.scores[sample_index].detach().cpu()
        top_scores, top_ids = torch.topk(scores, min(top_k, scores.shape[0]))
        predicted_answer_id = int(self.output.predicted_ids[sample_index].detach().cpu())
        predicted_answer = self.data.vocabulary.decode(predicted_answer_id)

        return VQAPrediction(
            sample_id=sample.sample_id,
            image=self.batch.images[sample_index],
            question=sample.question,
            predicted_answer_id=predicted_answer_id,
            predicted_answer=predicted_answer,
            predicted_score=float(self.output.confidence[sample_index].detach().cpu()),
            representative_answer=sample.answer,
            raw_answers=sample.raw_answers,
            vqa_target_score=float(self.batch.targets[sample_index, predicted_answer_id].detach().cpu()),
            top_predictions=tuple(
                RankedAnswerPrediction(
                    answer_id=int(answer_id),
                    answer=self.data.vocabulary.decode(int(answer_id)),
                    score=float(score),
                )
                for answer_id, score in zip(
                    top_ids.tolist(),
                    top_scores.tolist(),
                    strict=True,
                )
            ),
            source_index=sample.source.raw_index,
            image_id=sample.source.image_id,
        )

    def summary(self) -> dict[str, Any]:
        first = self.batch.samples[0]
        predicted_answers = [
            self.data.vocabulary.decode(index)
            for index in self.output.predicted_ids.detach().cpu().tolist()
        ]
        return {
            "device": str(self.device),
            "canonical_sample_count": len(self.data.complete_dataset.samples),
            "training_sample_count": len(self.data.training_dataset.samples),
            "validation_sample_count": len(self.data.validation_dataset.samples),
            "answer_vocabulary_size": len(self.data.vocabulary),
            "batch_size": len(self.batch.samples),
            "pixel_values_shape": tuple(self.batch.pixel_values.shape),
            "targets_shape": tuple(self.batch.targets.shape),
            "logits_shape": tuple(self.output.logits.shape),
            "predicted_ids": self.output.predicted_ids.detach().cpu().tolist(),
            "predicted_answers": predicted_answers,
            "max_answer_scores": self.output.confidence.detach().cpu().tolist(),
            "sample_id": first.sample_id,
            "question": first.question,
            "representative_answer": first.answer,
            "raw_answers": first.raw_answers,
            "source_index": first.source.raw_index,
            "image_id": first.source.image_id,
            "image_patch_embeddings_shape": tuple(self.output.image_patch_embeddings.shape),
            "text_token_embeddings_shape": tuple(self.output.text_token_embeddings.shape),
            "multimodal_embeddings_shape": tuple(self.output.multimodal_embeddings.shape),
            "fused_hidden_state_shape": tuple(self.output.fused_hidden_state.shape),
            "parameter_counts": self.parameter_counts,
        }


class ModalityContributionPipeline:
    def __init__(self, config: ModalityContributionPipelineConfig) -> None:
        self.config = config

    def run(self) -> ModalityContributionPipelineResult:
        set_seed(self.config.model.initialization_seed)
        device = resolve_device(self.config.device)
        data = VQADataPipeline(
            config=self.config.data,
            vision_model_name=self.config.model.vision.model_name,
        ).build()
        runtime_model_config = self.config.model.with_number_of_classes(
            len(data.vocabulary)
        )
        model = CompactViltModel(runtime_model_config.to_dict()).to(device)
        history = None
        if self.config.training is not None:
            trainer = VQATrainer(model=model, device=device, config=self.config.training)
            history = trainer.fit(data.training_loader, data.validation_loader)
        batch = next(iter(data.validation_loader)).to(device)
        model.eval()
        with torch.no_grad():
            output = model(
                pixel_values=batch.pixel_values,
                texts=batch.questions,
                return_tokens=self.config.return_tokens,
            )
        return ModalityContributionPipelineResult(
            data=data,
            batch=batch,
            output=output,
            model=model,
            parameter_counts=model.count_parameters(),
            device=device,
            training_history=history,
        )
