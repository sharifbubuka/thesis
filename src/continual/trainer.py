from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import ViltForQuestionAnswering


@dataclass(frozen=True, slots=True)
class TaskTrainingResult:
    losses: tuple[float, ...]


class ContinualVqaTrainer:
    def __init__(
        self,
        model: ViltForQuestionAnswering,
        *,
        device: torch.device,
        learning_rate: float,
        weight_decay: float,
        gradient_clip_norm: float,
    ) -> None:
        self.model = model
        self.device = device
        self.loss_function = nn.BCEWithLogitsLoss()
        # Resetting AdamW at each task boundary is intentional and reproducible.
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.gradient_clip_norm = gradient_clip_norm

    def fit(self, loader: DataLoader[Any], *, epochs: int) -> TaskTrainingResult:
        losses: list[float] = []
        for _ in range(epochs):
            self.model.train()
            total = 0.0
            batches = 0
            for batch in loader:
                inputs = {
                    key: value.to(self.device)
                    for key, value in batch["inputs"].items()
                }
                labels = batch["labels"].to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(**inputs, return_dict=True).logits
                loss = self.loss_function(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                self.optimizer.step()
                total += float(loss.detach().cpu())
                batches += 1
            losses.append(total / max(batches, 1))
        return TaskTrainingResult(tuple(losses))

    @torch.no_grad()
    def evaluate(self, loader: DataLoader[Any]) -> float:
        self.model.eval()
        correct = 0
        total = 0
        for batch in loader:
            inputs = {
                key: value.to(self.device)
                for key, value in batch["inputs"].items()
            }
            labels = batch["labels"].to(self.device)
            predictions = self.model(**inputs, return_dict=True).logits.argmax(dim=-1)
            rows = torch.arange(predictions.shape[0], device=self.device)
            correct += int((labels[rows, predictions] > 0).sum().item())
            total += predictions.shape[0]
        return correct / max(total, 1)
