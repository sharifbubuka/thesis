from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ContinualMetrics:
    average_accuracy: float
    average_forgetting: float
    backward_transfer: float
    per_task_forgetting: tuple[float, ...]


def performance_summary(matrix: np.ndarray) -> ContinualMetrics:
    """Summarize a square R[i,j] matrix, including the initial row R[0,j]."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1] + 1:
        raise ValueError("matrix must have shape (number_of_tasks + 1, number_of_tasks).")
    number_of_tasks = values.shape[1]
    final = values[-1]
    forgetting: list[float] = []
    backward_transfer: list[float] = []
    for task_index in range(number_of_tasks):
        learned_row = task_index + 1
        best_after_learning = float(np.nanmax(values[learned_row:, task_index]))
        forgetting.append(best_after_learning - float(final[task_index]))
        if task_index < number_of_tasks - 1:
            backward_transfer.append(float(final[task_index] - values[learned_row, task_index]))
    return ContinualMetrics(
        average_accuracy=float(np.nanmean(final)),
        average_forgetting=float(np.nanmean(forgetting)),
        backward_transfer=(
            float(np.nanmean(backward_transfer)) if backward_transfer else 0.0
        ),
        per_task_forgetting=tuple(forgetting),
    )
