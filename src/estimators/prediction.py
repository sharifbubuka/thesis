from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image


@dataclass(frozen=True, slots=True)
class RankedAnswerPrediction:
    answer_id: int
    answer: str
    score: float


@dataclass(frozen=True, slots=True)
class VQAPrediction:
    sample_id: str
    image: Image.Image
    question: str
    predicted_answer_id: int
    predicted_answer: str
    predicted_score: float
    representative_answer: str
    raw_answers: tuple[str, ...]
    vqa_target_score: float
    top_predictions: tuple[RankedAnswerPrediction, ...]
    source_index: int
    image_id: str

    @property
    def is_exact_representative_match(self) -> bool:
        return self.predicted_answer == self.representative_answer

    def summary(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "question": self.question,
            "predicted_answer": self.predicted_answer,
            "predicted_score": self.predicted_score,
            "representative_answer": self.representative_answer,
            "raw_answers": self.raw_answers,
            "vqa_target_score": self.vqa_target_score,
            "is_exact_representative_match": self.is_exact_representative_match,
            "source_index": self.source_index,
            "image_id": self.image_id,
            "top_predictions": [
                {
                    "answer_id": item.answer_id,
                    "answer": item.answer,
                    "score": item.score,
                }
                for item in self.top_predictions
            ],
        }
