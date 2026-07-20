from __future__ import annotations

from typing import NamedTuple

import torch


class CompactViltInput(NamedTuple):
    """
    Inputs passed to the multimodal model.
    """

    pixel_values: torch.Tensor
    input_ids: torch.Tensor
    text_attention_mask: torch.Tensor


class CompactViltOutput(NamedTuple):
    """
    Outputs returned by the complete multimodal model.
    """

    # Final prediction outputs
    logits: torch.Tensor
    probabilities: torch.Tensor
    predicted_class_ids: torch.Tensor
    confidence: torch.Tensor
    entropy: torch.Tensor

    # Raw encoder outputs
    image_patch_embeddings: torch.Tensor
    text_token_embeddings: torch.Tensor

    # Projected modality representations
    projected_image_embeddings: torch.Tensor
    projected_text_embeddings: torch.Tensor

    # Multimodal representations
    fused_hidden_state: torch.Tensor
    pooled_output: torch.Tensor

    # Optional intermediate encoder states
    vision_hidden_states: tuple[torch.Tensor, ...] | None
    text_hidden_states: tuple[torch.Tensor, ...] | None
    fusion_hidden_states: tuple[torch.Tensor, ...] | None

    # Optional attention maps
    vision_attentions: tuple[torch.Tensor, ...] | None
    text_attentions: tuple[torch.Tensor, ...] | None
    fusion_attentions: tuple[torch.Tensor, ...] | None

    # Masks
    image_attention_mask: torch.Tensor
    text_attention_mask: torch.Tensor
    multimodal_attention_mask: torch.Tensor

    # Sequence boundaries
    multimodal_cls_index: int | None
    image_token_start: int
    image_token_end: int
    text_token_start: int
    text_token_end: int

    # Image metadata
    patch_grid_size: tuple[int, int]

    # Optional text metadata
    text_tokens: tuple[tuple[str, ...], ...] | None
    
class VisionEncoderOutput(NamedTuple):
    """
    Outputs returned by the ViT vision encoder.

    Shapes:
        patch_embeddings:
            [batch_size, number_of_patches, hidden_size]

        cls_embedding:
            [batch_size, hidden_size], or None when the original
            ViT CLS token is discarded without being returned.

        attention_mask:
            [batch_size, number_of_patches]

        hidden_states:
            Tuple containing one tensor per embedding/encoder stage.
            Each tensor normally has shape:
            [batch_size, number_of_patches + 1, hidden_size]

        attentions:
            Tuple containing one tensor per encoder layer.
            Each tensor normally has shape:
            [batch_size, number_of_heads, sequence_length, sequence_length]
    """

    patch_embeddings: torch.Tensor
    cls_embedding: torch.Tensor | None
    attention_mask: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] | None
    attentions: tuple[torch.Tensor, ...] | None
    patch_grid_size: tuple[int, int]
    
class TextEncoderOutput(NamedTuple):
    """
    Outputs returned by the Ettin text encoder.

    Shapes:
        token_embeddings:
            [batch_size, sequence_length, hidden_size]

        attention_mask:
            [batch_size, sequence_length]

        input_ids:
            [batch_size, sequence_length]

        hidden_states:
            One tensor for the embedding layer and each encoder layer.

        attentions:
            One tensor per encoder layer with shape:
            [batch_size, heads, sequence_length, sequence_length]
    """

    token_embeddings: torch.Tensor
    attention_mask: torch.Tensor
    input_ids: torch.Tensor
    tokens: tuple[tuple[str, ...], ...] | None
    hidden_states: tuple[torch.Tensor, ...] | None
    attentions: tuple[torch.Tensor, ...] | None
    
class ProjectionOutput(NamedTuple):
    """
    Output returned by a modality projection layer.

    Shapes:
        embeddings:
            [batch_size, sequence_length, output_size]

        attention_mask:
            [batch_size, sequence_length]
    """

    embeddings: torch.Tensor
    attention_mask: torch.Tensor