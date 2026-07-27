import torch

from src.vilt.outputs import PredictionResult


def test_result_summary_is_json_serialisable_shape_metadata() -> None:
    result = PredictionResult(
        sample_id="sample",
        benchmark="vqav2",
        checkpoint="checkpoint",
        predicted_label="2",
        predicted_id=2,
        target_label="2",
        is_correct=True,
        logits=torch.tensor([[0.1, 0.2]]),
        scores=torch.tensor([[0.5, 0.6]]),
        input_shapes={"input_ids": (1, 5)},
        hidden_states=(torch.zeros(1, 5, 4),),
        attentions=(torch.zeros(1, 2, 5, 5),),
        tokens=("[CLS]", "how"),
        metadata={},
    )
    summary = result.summary()
    assert summary["logits_shape"] == [1, 2]
    assert summary["forward_pass_valid"] is True
