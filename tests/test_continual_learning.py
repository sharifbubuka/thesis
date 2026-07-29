from __future__ import annotations

import numpy as np
import pytest

from src.continual.metrics import performance_summary
from src.continual.registry import CONTINUAL_TASKS
from src.continual.vocabulary import AnswerVocabulary, normalize_answer


def test_continual_task_order_is_fixed() -> None:
    assert [spec.name.value for spec in CONTINUAL_TASKS] == ["textvqa", "gqa", "vqav2"]


def test_shared_vocabulary_is_deterministic_and_normalized() -> None:
    vocabulary = AnswerVocabulary.build(
        [["The cat", "cat", "YES"], ["yes", "red"]],
        max_size=4,
    )
    assert vocabulary.answers == ("<unk>", "cat", "yes", "red")
    assert vocabulary.encode("A cat!") == vocabulary.encode("cat")
    assert vocabulary.encode("not present") == 0
    assert normalize_answer("  The, RED!  ") == "red"
    assert AnswerVocabulary.from_dict(vocabulary.to_dict()) == vocabulary


def test_continual_metrics_use_the_same_evolving_model_rows() -> None:
    matrix = np.array(
        [
            [0.10, 0.10, 0.10],
            [0.80, 0.15, 0.10],
            [0.65, 0.75, 0.12],
            [0.60, 0.70, 0.85],
        ]
    )
    metrics = performance_summary(matrix)
    assert metrics.average_accuracy == pytest.approx((0.60 + 0.70 + 0.85) / 3)
    assert metrics.per_task_forgetting == pytest.approx((0.20, 0.05, 0.0))
    assert metrics.backward_transfer == pytest.approx((-0.20 - 0.05) / 2)


def test_performance_matrix_shape_is_validated() -> None:
    with pytest.raises(ValueError, match="shape"):
        performance_summary(np.zeros((3, 3)))
