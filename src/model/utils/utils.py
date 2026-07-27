from __future__ import annotations

import torch
from torch import Tensor

from collections import Counter
from collections.abc import Callable, Mapping, Sequence


def compute_prediction_statistics(
    logits: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Convert classifier logits into prediction statistics.

    Args:
        logits:
            Tensor with shape [batch_size, number_of_classes].

    Returns:
        probabilities:
            Softmax probabilities with shape
            [batch_size, number_of_classes].

        predicted_class_ids:
            Predicted class IDs with shape [batch_size].

        confidence:
            Probability assigned to each predicted class,
            with shape [batch_size].

        entropy:
            Predictive entropy for each sample,
            with shape [batch_size].
    """

    if logits.ndim != 2:
        raise ValueError(
            "logits must have shape [batch_size, number_of_classes], "
            f"but received {tuple(logits.shape)}."
        )

    probabilities = torch.softmax(logits, dim=-1)

    confidence, predicted_class_ids = probabilities.max(dim=-1)

    safe_probabilities = probabilities.clamp_min(
        torch.finfo(probabilities.dtype).tiny
    )

    entropy = -torch.sum(
        probabilities * torch.log(safe_probabilities),
        dim=-1,
    )

    return (
        probabilities,
        predicted_class_ids,
        confidence,
        entropy,
    )
    
def calculate_sequence_boundaries(
    *,
    number_of_image_tokens: int,
    number_of_text_tokens: int,
    use_cls_token: bool,
) -> dict[str, int | None]:
    """
    Calculate the modality positions in the fused sequence.

    All end positions are exclusive.
    """

    if number_of_image_tokens <= 0:
        raise ValueError(
            "number_of_image_tokens must be positive."
        )

    if number_of_text_tokens <= 0:
        raise ValueError(
            "number_of_text_tokens must be positive."
        )

    cls_offset = 1 if use_cls_token else 0

    image_start = cls_offset
    image_end = image_start + number_of_image_tokens

    text_start = image_end
    text_end = text_start + number_of_text_tokens

    return {
        "multimodal_cls_index": 0 if use_cls_token else None,
        "image_token_start": image_start,
        "image_token_end": image_end,
        "text_token_start": text_start,
        "text_token_end": text_end,
    }
    
def build_image_attention_mask(
    *,
    batch_size: int,
    number_of_image_tokens: int,
    device: torch.device,
) -> Tensor:
    """
    Create an attention mask for image patch tokens.

    All image patches are considered valid.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if number_of_image_tokens <= 0:
        raise ValueError(
            "number_of_image_tokens must be positive."
        )

    return torch.ones(
        batch_size,
        number_of_image_tokens,
        dtype=torch.long,
        device=device,
    )
    
def build_multimodal_attention_mask(
    *,
    image_attention_mask: Tensor,
    text_attention_mask: Tensor,
    use_cls_token: bool,
) -> Tensor:
    """
    Combine image and text masks into one multimodal attention mask.

    Sequence order:
        [CLS] [IMAGE] [TEXT]
    """

    if image_attention_mask.ndim != 2:
        raise ValueError(
            "image_attention_mask must have shape "
            "[batch_size, number_of_image_tokens]."
        )

    if text_attention_mask.ndim != 2:
        raise ValueError(
            "text_attention_mask must have shape "
            "[batch_size, number_of_text_tokens]."
        )

    if image_attention_mask.shape[0] != text_attention_mask.shape[0]:
        raise ValueError(
            "Image and text masks must have the same batch size."
        )

    image_attention_mask = image_attention_mask.to(
        dtype=torch.long
    )
    text_attention_mask = text_attention_mask.to(
        device=image_attention_mask.device,
        dtype=torch.long,
    )

    masks = []

    if use_cls_token:
        cls_mask = torch.ones(
            image_attention_mask.shape[0],
            1,
            dtype=torch.long,
            device=image_attention_mask.device,
        )
        masks.append(cls_mask)

    masks.extend(
        [
            image_attention_mask,
            text_attention_mask,
        ]
    )

    return torch.cat(masks, dim=1)

AnswerNormalizer = Callable[[str], str]

def build_vqa_soft_target(
    *,
    answers: Sequence[str],
    answer_to_id: Mapping[str, int],
    number_of_classes: int,
    normalize_answer: AnswerNormalizer,
) -> Tensor:
    """
    Build a soft VQA target from multiple annotator answers.

    Every vocabulary answer receives:

        min(normalized_answer_count / 3, 1)

    Answers absent from the fixed vocabulary are ignored.
    """

    if number_of_classes <= 1:
        raise ValueError(
            "number_of_classes must be greater than one."
        )

    if not answers:
        raise ValueError(
            "At least one annotator answer is required."
        )

    normalized_answers = [
        normalize_answer(answer)
        for answer in answers
        if answer.strip()
    ]

    if not normalized_answers:
        raise ValueError(
            "No valid annotator answers remain after normalization."
        )

    answer_counts = Counter(normalized_answers)

    target = torch.zeros(
        number_of_classes,
        dtype=torch.float32,
    )

    for answer, count in answer_counts.items():
        answer_id = answer_to_id.get(answer)

        if answer_id is None:
            continue

        if not 0 <= answer_id < number_of_classes:
            raise ValueError(
                f"Answer ID {answer_id} for '{answer}' is "
                "outside the configured vocabulary."
            )

        target[answer_id] = min(
            count / 3.0,
            1.0,
        )

    return target