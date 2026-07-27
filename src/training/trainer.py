from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.contracts import VQABatch
from src.model import CompactViltModel


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0


@dataclass(slots=True)
class TrainingHistory:
    training_loss: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)
    validation_vqa_score: list[float] = field(default_factory=list)


class VQATrainer:
    def __init__(
        self,
        *,
        model: CompactViltModel,
        device: torch.device,
        config: TrainingConfig,
    ) -> None:
        self.model = model
        self.device = device
        self.config = config
        self.loss_function = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    def fit(
        self,
        training_loader: DataLoader[VQABatch],
        validation_loader: DataLoader[VQABatch],
    ) -> TrainingHistory:
        history = TrainingHistory()
        for _ in range(self.config.epochs):
            history.training_loss.append(self._train_epoch(training_loader))
            validation_loss, validation_score = self.evaluate(validation_loader)
            history.validation_loss.append(validation_loss)
            history.validation_vqa_score.append(validation_score)
        return history

    def _train_epoch(self, loader: DataLoader[VQABatch]) -> float:
        self.model.train()
        total_loss = 0.0
        number_of_batches = 0
        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            output = self.model(
                pixel_values=batch.pixel_values,
                texts=batch.questions,
                return_tokens=False,
            )
            loss = self.loss_function(output.logits, batch.targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_norm,
            )
            self.optimizer.step()
            total_loss += float(loss.detach().cpu())
            number_of_batches += 1
        return total_loss / max(number_of_batches, 1)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader[VQABatch]) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_score = 0.0
        number_of_samples = 0
        number_of_batches = 0
        for batch in loader:
            batch = batch.to(self.device)
            output = self.model(
                pixel_values=batch.pixel_values,
                texts=batch.questions,
                return_tokens=False,
            )
            total_loss += float(self.loss_function(output.logits, batch.targets).cpu())
            row_indices = torch.arange(output.predicted_ids.shape[0], device=self.device)
            total_score += float(batch.targets[row_indices, output.predicted_ids].sum().cpu())
            number_of_samples += len(batch.samples)
            number_of_batches += 1
        return (
            total_loss / max(number_of_batches, 1),
            total_score / max(number_of_samples, 1),
        )
