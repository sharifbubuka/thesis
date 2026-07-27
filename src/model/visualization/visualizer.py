from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA

from src.data.answers import AnswerVocabulary
from src.estimators.prediction import VQAPrediction
from src.model.contracts import CompactViltOutput


class VQAModelVisualizer:
    def plot_top_predictions(
        self,
        output: CompactViltOutput,
        vocabulary: AnswerVocabulary,
        sample_index: int = 0,
        top_k: int = 10,
    ) -> None:
        scores = output.scores[sample_index].detach().cpu()
        top_scores, top_ids = torch.topk(scores, min(top_k, scores.shape[0]))
        labels = [vocabulary.decode(index) for index in top_ids.tolist()]
        figure, axis = plt.subplots(figsize=(10, 6))
        axis.barh(list(reversed(labels)), list(reversed(top_scores.tolist())))
        axis.set_xlim(0.0, 1.0)
        axis.set_xlabel("Sigmoid answer score")
        axis.set_title("Top predicted answers")
        figure.tight_layout()
        plt.show()

    def plot_patch_norms(self, output: CompactViltOutput, sample_index: int = 0) -> None:
        grid_height, grid_width = output.patch_grid_size
        values = output.image_patch_embeddings[sample_index].detach().cpu().norm(dim=-1)
        figure, axis = plt.subplots(figsize=(7, 6))
        heatmap = axis.imshow(values.reshape(grid_height, grid_width).numpy())
        axis.set_title("Image patch embedding norms")
        axis.axis("off")
        figure.colorbar(heatmap, ax=axis)
        figure.tight_layout()
        plt.show()

    def plot_fused_token_projection(
        self, output: CompactViltOutput, sample_index: int = 0
    ) -> None:
        embeddings = output.fused_hidden_state[sample_index].detach().cpu().numpy()
        projection = PCA(n_components=2).fit_transform(embeddings)
        figure, axis = plt.subplots(figsize=(8, 7))
        axis.scatter(
            projection[output.image_token_start:output.image_token_end, 0],
            projection[output.image_token_start:output.image_token_end, 1],
            label="Image patches",
            alpha=0.65,
        )
        axis.scatter(
            projection[output.text_token_start:output.text_token_end, 0],
            projection[output.text_token_start:output.text_token_end, 1],
            label="Text tokens",
            alpha=0.85,
        )
        axis.set_title("PCA of fused multimodal tokens")
        axis.legend()
        figure.tight_layout()
        plt.show()

    def plot_prediction(self, prediction: VQAPrediction) -> None:
        """Display a validation image together with its decoded prediction."""
        labels = [item.answer for item in reversed(prediction.top_predictions)]
        values = [item.score for item in reversed(prediction.top_predictions)]

        figure, axes = plt.subplots(1, 2, figsize=(15, 6))
        axes[0].imshow(prediction.image)
        axes[0].axis("off")
        axes[0].set_title(
            f"Question: {prediction.question}\n"
            f"Prediction: {prediction.predicted_answer} "
            f"({prediction.predicted_score:.3f})\n"
            f"Reference: {prediction.representative_answer}\n"
            f"VQA target score: {prediction.vqa_target_score:.3f}"
        )

        axes[1].barh(labels, values)
        axes[1].set_xlim(0.0, 1.0)
        axes[1].set_xlabel("Sigmoid answer score")
        axes[1].set_title("Top predicted answers")

        figure.tight_layout()
        plt.show()
