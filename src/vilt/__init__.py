from __future__ import annotations

from typing import Any

__all__ = [
    "PredictionResult",
    "ViltAdapter",
    "assert_stage_one_result",
    "create_adapter",
    "load_vilt_checkpoint",
    "model_parameter_count",
    "save_stage_one_report",
]


def __getattr__(name: str) -> Any:
    if name == "PredictionResult":
        from src.vilt.outputs import PredictionResult
        return PredictionResult
    if name in {"ViltAdapter", "create_adapter"}:
        from src.vilt.adapters import ViltAdapter, create_adapter
        return {"ViltAdapter": ViltAdapter, "create_adapter": create_adapter}[name]
    if name == "load_vilt_checkpoint":
        from src.vilt.loader import load_vilt_checkpoint
        return load_vilt_checkpoint
    if name in {"assert_stage_one_result", "model_parameter_count", "save_stage_one_report"}:
        from src.vilt.inspector import (
            assert_stage_one_result,
            model_parameter_count,
            save_stage_one_report,
        )
        return {
            "assert_stage_one_result": assert_stage_one_result,
            "model_parameter_count": model_parameter_count,
            "save_stage_one_report": save_stage_one_report,
        }[name]
    raise AttributeError(name)
