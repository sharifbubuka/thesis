from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import ViltForQuestionAnswering, ViltProcessor

from src.continual.data import load_vqa_samples, make_loader
from src.continual.metrics import ContinualMetrics, performance_summary
from src.continual.registry import CONTINUAL_TASKS
from src.continual.trainer import ContinualVqaTrainer
from src.continual.vocabulary import AnswerVocabulary
from src.data.contracts import CanonicalSample
from src.utils.reproducibility import set_seed


@dataclass(frozen=True, slots=True)
class ContinualExperimentConfig:
    base_checkpoint: str = "dandelin/vilt-b32-mlm"
    train_samples_per_task: int = 1_000
    validation_samples_per_task: int = 250
    vocabulary_size: int = 5_000
    batch_size: int = 8
    epochs_per_task: int = 3
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    seed: int = 42
    save_stage_checkpoints: bool = True


class ContinualExperiment:
    """Train one ViLT VQA model through TextVQA, GQA, and VQAv2."""

    def __init__(
        self,
        config: ContinualExperimentConfig,
        *,
        device: torch.device,
        output_dir: Path,
    ) -> None:
        self.config = config
        self.device = device
        self.output_dir = output_dir

    def _load_data(
        self,
    ) -> tuple[dict[str, list[CanonicalSample]], dict[str, list[CanonicalSample]]]:
        training: dict[str, list[CanonicalSample]] = {}
        validation: dict[str, list[CanonicalSample]] = {}
        for offset, spec in enumerate(CONTINUAL_TASKS):
            training[spec.name.value] = load_vqa_samples(
                spec,
                training=True,
                count=self.config.train_samples_per_task,
                seed=self.config.seed + offset,
            )
            validation[spec.name.value] = load_vqa_samples(
                spec,
                training=False,
                count=self.config.validation_samples_per_task,
                seed=self.config.seed + offset,
            )
        return training, validation

    def _build_vocabulary(
        self, training: dict[str, list[CanonicalSample]]
    ) -> AnswerVocabulary:
        answer_groups = (
            tuple(str(answer) for answer in sample.metadata.get("answers", ()))
            for task_samples in training.values()
            for sample in task_samples
        )
        return AnswerVocabulary.build(answer_groups, max_size=self.config.vocabulary_size)

    def _initialize_model(
        self, vocabulary: AnswerVocabulary
    ) -> tuple[ViltProcessor, ViltForQuestionAnswering]:
        id2label = {index: answer for index, answer in enumerate(vocabulary.answers)}
        label2id = {answer: index for index, answer in id2label.items()}
        processor = ViltProcessor.from_pretrained(self.config.base_checkpoint)
        model = ViltForQuestionAnswering.from_pretrained(
            self.config.base_checkpoint,
            num_labels=len(vocabulary.answers),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
            attn_implementation="eager",
        )
        model.to(self.device)
        return processor, model

    def run(self) -> tuple[pd.DataFrame, ContinualMetrics]:
        set_seed(self.config.seed)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        training, validation = self._load_data()
        vocabulary = self._build_vocabulary(training)
        processor, model = self._initialize_model(vocabulary)
        max_length = model.config.max_position_embeddings

        validation_loaders = {
            task: make_loader(
                samples,
                processor=processor,
                vocabulary=vocabulary,
                max_length=max_length,
                batch_size=self.config.batch_size,
                shuffle=False,
            )
            for task, samples in validation.items()
        }
        task_names = [spec.name.value for spec in CONTINUAL_TASKS]
        matrix = np.zeros((len(task_names) + 1, len(task_names)), dtype=float)

        evaluator = ContinualVqaTrainer(
            model,
            device=self.device,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            gradient_clip_norm=self.config.gradient_clip_norm,
        )
        matrix[0] = [evaluator.evaluate(validation_loaders[name]) for name in task_names]

        training_history: dict[str, list[float]] = {}
        for task_index, task_name in enumerate(task_names, start=1):
            training_loader = make_loader(
                training[task_name],
                processor=processor,
                vocabulary=vocabulary,
                max_length=max_length,
                batch_size=self.config.batch_size,
                shuffle=True,
            )
            # A fresh optimizer at each boundary avoids carrying task-specific moments.
            trainer = ContinualVqaTrainer(
                model,
                device=self.device,
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                gradient_clip_norm=self.config.gradient_clip_norm,
            )
            result = trainer.fit(training_loader, epochs=self.config.epochs_per_task)
            training_history[task_name] = list(result.losses)
            matrix[task_index] = [
                trainer.evaluate(validation_loaders[name]) for name in task_names
            ]
            if self.config.save_stage_checkpoints:
                model.save_pretrained(self.output_dir / f"after_{task_name}")

        dataframe = pd.DataFrame(
            matrix,
            columns=task_names,
            index=["initial", *(f"after_{name}" for name in task_names)],
        )
        metrics = performance_summary(matrix)
        dataframe.to_csv(self.output_dir / "performance_matrix.csv", index_label="model_state")
        with (self.output_dir / "answer_vocabulary.json").open("w", encoding="utf-8") as handle:
            json.dump(vocabulary.to_dict(), handle, indent=2)
        with (self.output_dir / "experiment.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "config": asdict(self.config),
                    "task_order": task_names,
                    "training_loss": training_history,
                    "metrics": asdict(metrics),
                },
                handle,
                indent=2,
            )
        processor.save_pretrained(self.output_dir / "processor")
        return dataframe, metrics
