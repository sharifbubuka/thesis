from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch

from src.vilt.outputs import PredictionResult


def model_parameter_count(model: torch.nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable}


def assert_stage_one_result(result: PredictionResult) -> None:
    assert result.logits.ndim >= 1
    assert torch.isfinite(result.logits).all()
    assert result.hidden_states is not None
    assert result.attentions is not None
    assert len(result.hidden_states) >= 2
    assert len(result.attentions) >= 1


def save_stage_one_report(results: Iterable[PredictionResult], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [result.summary() for result in results]
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path
