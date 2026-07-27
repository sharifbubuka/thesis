from __future__ import annotations

import matplotlib.pyplot as plt

from .trainer import TrainingHistory


class TrainingVisualizer:
    def plot(self, history: TrainingHistory) -> None:
        epochs = list(range(1, len(history.training_loss) + 1))
        figure, axis = plt.subplots(figsize=(10, 5))
        axis.plot(epochs, history.training_loss, label="Training loss")
        axis.plot(epochs, history.validation_loss, label="Validation loss")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("BCE loss")
        axis.set_title("Training and validation loss")
        axis.legend()
        figure.tight_layout()
        plt.show()

        figure, axis = plt.subplots(figsize=(10, 5))
        axis.plot(epochs, history.validation_vqa_score, label="Validation VQA score")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Mean soft VQA score")
        axis.set_ylim(0.0, 1.0)
        axis.set_title("Validation VQA score")
        axis.legend()
        figure.tight_layout()
        plt.show()
