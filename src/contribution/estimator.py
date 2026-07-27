from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from src.benchmarks.contracts import MultimodalSample
from src.contribution.ablation import (
    ablated_sample,
    inclusion_exclusion,
    normalized_absolute_shares,
)
from src.contribution.contracts import ContributionResult, TargetSpec
from src.contribution.gradients import ViltActivationCapture, attribution_scores
from src.vilt.adapters import ViltAdapter


class ModalityContributionEstimator:
    def __init__(self, adapter: ViltAdapter) -> None:
        self.adapter = adapter

    def _target(self, logits: torch.Tensor, target: TargetSpec) -> torch.Tensor:
        return self.adapter.select_target_score(logits, target.target_id)

    def _score(self, sample: MultimodalSample, target: TargetSpec) -> float:
        inputs = self.adapter.prepare_inputs(sample)
        with torch.inference_mode():
            outputs = self.adapter.model(**inputs, return_dict=True)
        return float(self._target(outputs.logits, target).detach().cpu().item())

    def _prediction_statistics(
        self, logits: torch.Tensor, predicted_id: int | None
    ) -> tuple[float, float, float]:
        probabilities = self.adapter.probabilities(logits)
        flat = probabilities.reshape(-1)
        if predicted_id is None:
            confidence = float(flat[0].item())
            return confidence, abs(confidence - 0.5), self._binary_entropy(confidence)

        confidence = float(flat[predicted_id].item())
        if flat.numel() > 1:
            top2 = torch.topk(flat, k=min(2, flat.numel())).values
            margin = float((top2[0] - top2[1]).item()) if top2.numel() == 2 else confidence
        else:
            margin = confidence
        if self.adapter.spec.benchmark.value == "vqav2":
            safe = flat.clamp(1e-12, 1 - 1e-12)
            entropy = float(
                (-(safe * safe.log() + (1 - safe) * (1 - safe).log())).mean().item()
            )
        else:
            safe = flat.clamp_min(1e-12)
            entropy = float((-(safe * safe.log()).sum()).item())
        return confidence, margin, entropy

    @staticmethod
    def _binary_entropy(probability: float) -> float:
        probability = min(max(probability, 1e-12), 1 - 1e-12)
        return -(probability * math.log(probability) + (1 - probability) * math.log(1 - probability))

    def estimate(self, sample: MultimodalSample) -> ContributionResult:
        self.adapter._validate_sample(sample)
        inputs = self.adapter.prepare_inputs(sample)
        self.adapter.model.zero_grad(set_to_none=True)

        with ViltActivationCapture(self.adapter.model) as capture:
            outputs = self.adapter.model(
                **inputs,
                output_hidden_states=True,
                output_attentions=False,
                return_dict=True,
            )
            logits = outputs.logits
            predicted_label, predicted_id, _ = self.adapter._decode(logits, sample)
            target = TargetSpec(predicted_id, predicted_label)
            target_score = self._target(logits, target)
            target_score.backward()

        image_grad_norm, image_gx, patch_scores = attribution_scores(capture.values.image)
        text_grad_norm, text_gx, token_scores = attribution_scores(capture.values.text)

        full_score = float(target_score.detach().cpu().item())
        image_only = self._score(ablated_sample(sample, remove_text=True), target)
        text_only = self._score(ablated_sample(sample, remove_images=True), target)
        baseline = self._score(
            ablated_sample(sample, remove_images=True, remove_text=True), target
        )
        image_contrib, text_contrib, interaction, total = inclusion_exclusion(
            full=full_score,
            image_only=image_only,
            text_only=text_only,
            baseline=baseline,
        )
        image_share, text_share, interaction_share = normalized_absolute_shares(
            image_contrib, text_contrib, interaction
        )

        confidence, margin, entropy = self._prediction_statistics(logits.detach(), predicted_id)
        target_label = None if sample.target is None else str(sample.target)
        is_correct = None if target_label is None else predicted_label.lower() == target_label.lower()
        tokens = tuple(self.adapter.processor.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))

        usable_token_scores = token_scores[: len(tokens)] if token_scores.numel() else token_scores
        top_token_index = int(usable_token_scores.argmax().item()) if usable_token_scores.numel() else None
        top_token = tokens[top_token_index] if top_token_index is not None else None
        top_token_score = (
            float(usable_token_scores[top_token_index].item()) if top_token_index is not None else None
        )
        top_patch_index = int(patch_scores.argmax().item()) if patch_scores.numel() else None
        top_patch_score = (
            float(patch_scores[top_patch_index].item()) if top_patch_index is not None else None
        )

        cls_embedding = self._extract_cls(outputs.hidden_states)
        return ContributionResult(
            sample_id=sample.sample_id,
            benchmark=self.adapter.spec.benchmark.value,
            checkpoint=self.adapter.spec.checkpoint,
            prediction=predicted_label,
            predicted_id=predicted_id,
            target_label=target_label,
            is_correct=is_correct,
            confidence=confidence,
            margin=margin,
            entropy=entropy,
            full_score=full_score,
            image_only_score=image_only,
            text_only_score=text_only,
            baseline_score=baseline,
            image_contribution=image_contrib,
            text_contribution=text_contrib,
            interaction_contribution=interaction,
            total_effect=total,
            image_share=image_share,
            text_share=text_share,
            interaction_share=interaction_share,
            image_gradient_norm=image_grad_norm,
            text_gradient_norm=text_grad_norm,
            image_gradient_x_input=image_gx,
            text_gradient_x_input=text_gx,
            top_patch_index=top_patch_index,
            top_patch_score=top_patch_score,
            top_token=top_token,
            top_token_index=top_token_index,
            top_token_score=top_token_score,
            cls_embedding=cls_embedding,
            image_patch_scores=patch_scores.numpy() if patch_scores.numel() else None,
            text_token_scores=usable_token_scores.numpy() if usable_token_scores.numel() else None,
            tokens=tokens,
            metadata=dict(sample.metadata),
        )

    @staticmethod
    def _extract_cls(hidden_states: Any) -> np.ndarray | None:
        if hidden_states is None:
            return None

        def final_cls_vectors(values: Any) -> list[torch.Tensor]:
            if isinstance(values, torch.Tensor):
                return [values[:, 0, :].mean(dim=0)]
            if isinstance(values, (tuple, list)) and values:
                # A flat tuple of tensors is the layer sequence for one encoder run.
                if all(isinstance(item, torch.Tensor) for item in values):
                    final_layer = values[-1]
                    return [final_layer[:, 0, :].mean(dim=0)]
                vectors: list[torch.Tensor] = []
                for item in values:
                    vectors.extend(final_cls_vectors(item))
                return vectors
            return []

        vectors = final_cls_vectors(hidden_states)
        if not vectors:
            return None
        return torch.stack(vectors).mean(dim=0).detach().cpu().float().numpy()
