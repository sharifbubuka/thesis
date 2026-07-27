import pytest

from src.benchmarks.registry import BenchmarkName, get_checkpoint_spec


def test_all_stage_one_checkpoints_are_registered() -> None:
    assert get_checkpoint_spec(BenchmarkName.VQAV2).expected_images == 1
    assert get_checkpoint_spec(BenchmarkName.NLVR2).expected_images == 2
    assert get_checkpoint_spec(BenchmarkName.COCO_RETRIEVAL).expected_images == 1


def test_unknown_benchmark_has_helpful_error() -> None:
    with pytest.raises(ValueError, match="Supported"):
        get_checkpoint_spec("unknown")
