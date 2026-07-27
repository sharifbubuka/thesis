from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from ..contracts import (MultimodalSequenceOutput)
from ..utils import (
    build_multimodal_attention_mask,
    calculate_sequence_boundaries,
)


class MultimodalSequenceBuilder(nn.Module):
    """
    Combine projected image and text embeddings into one sequence.

    Sequence layout:
        [MM-CLS] [IMAGE PATCHES] [TEXT TOKENS]
    """

    CLS_MODALITY_ID = 0
    IMAGE_MODALITY_ID = 1
    TEXT_MODALITY_ID = 2

    NUMBER_OF_MODALITY_TYPES = 3

    def __init__(
        self,
        config: dict[str, Any],
    ) -> None:
        super().__init__()

        self.config = config

        self.projection_config = config["projection"]
        self.sequence_config = config["multimodal_sequence"]

        self.hidden_size: int = self.projection_config[
            "hidden_size"
        ]

        self.use_cls_token: bool = self.sequence_config[
            "use_cls_token"
        ]

        self.use_modality_embeddings: bool = (
            self.sequence_config["use_modality_embeddings"]
        )

        self.dropout_probability: float = self.sequence_config[
            "dropout"
        ]

        self.use_layer_norm: bool = self.sequence_config[
            "use_layer_norm"
        ]

        self._validate_config()

        self.cls_token: nn.Parameter | None

        if self.use_cls_token:
            self.cls_token = nn.Parameter(
                torch.empty(
                    1,
                    1,
                    self.hidden_size,
                )
            )
        else:
            self.register_parameter(
                "cls_token",
                None,
            )

        self.modality_embeddings: nn.Embedding | None

        if self.use_modality_embeddings:
            self.modality_embeddings = nn.Embedding(
                num_embeddings=self.NUMBER_OF_MODALITY_TYPES,
                embedding_dim=self.hidden_size,
            )
        else:
            self.modality_embeddings = None

        if self.use_layer_norm:
            self.layer_norm: nn.Module = nn.LayerNorm(
                normalized_shape=self.hidden_size
            )
        else:
            self.layer_norm = nn.Identity()

        self.dropout = nn.Dropout(
            p=self.dropout_probability
        )

        self.reset_parameters()

    def _validate_config(self) -> None:
        required_fields = {
            "use_cls_token",
            "use_modality_embeddings",
            "dropout",
            "use_layer_norm",
        }

        missing_fields = (
            required_fields - self.sequence_config.keys()
        )

        if missing_fields:
            raise ValueError(
                "Missing multimodal sequence configuration "
                f"fields: {sorted(missing_fields)}"
            )

        if self.hidden_size <= 0:
            raise ValueError(
                "The multimodal hidden size must be positive."
            )

        if not 0.0 <= self.dropout_probability < 1.0:
            raise ValueError(
                "Multimodal sequence dropout must be in "
                "the range [0, 1)."
            )

    def reset_parameters(self) -> None:
        """
        Initialize newly learned multimodal parameters.
        """

        if self.cls_token is not None:
            nn.init.normal_(
                self.cls_token,
                mean=0.0,
                std=0.02,
            )

        if self.modality_embeddings is not None:
            nn.init.normal_(
                self.modality_embeddings.weight,
                mean=0.0,
                std=0.02,
            )

    def forward(
        self,
        *,
        image_embeddings: Tensor,
        image_attention_mask: Tensor,
        text_embeddings: Tensor,
        text_attention_mask: Tensor,
    ) -> MultimodalSequenceOutput:
        """
        Construct the combined multimodal sequence.

        Args:
            image_embeddings:
                [batch_size, image_tokens, hidden_size]

            image_attention_mask:
                [batch_size, image_tokens]

            text_embeddings:
                [batch_size, text_tokens, hidden_size]

            text_attention_mask:
                [batch_size, text_tokens]
        """

        self._validate_inputs(
            image_embeddings=image_embeddings,
            image_attention_mask=image_attention_mask,
            text_embeddings=text_embeddings,
            text_attention_mask=text_attention_mask,
        )

        batch_size = image_embeddings.shape[0]
        number_of_image_tokens = image_embeddings.shape[1]
        number_of_text_tokens = text_embeddings.shape[1]

        image_embeddings = self._add_modality_embedding(
            embeddings=image_embeddings,
            modality_id=self.IMAGE_MODALITY_ID,
        )

        text_embeddings = self._add_modality_embedding(
            embeddings=text_embeddings,
            modality_id=self.TEXT_MODALITY_ID,
        )

        sequence_parts: list[Tensor] = []

        if self.cls_token is not None:
            cls_embeddings = self.cls_token.expand(
                batch_size,
                -1,
                -1,
            )

            cls_embeddings = self._add_modality_embedding(
                embeddings=cls_embeddings,
                modality_id=self.CLS_MODALITY_ID,
            )

            sequence_parts.append(cls_embeddings)

        sequence_parts.extend(
            [
                image_embeddings,
                text_embeddings,
            ]
        )

        multimodal_embeddings = torch.cat(
            sequence_parts,
            dim=1,
        )

        multimodal_embeddings = self.layer_norm(
            multimodal_embeddings
        )

        multimodal_embeddings = self.dropout(
            multimodal_embeddings
        )

        multimodal_attention_mask = (
            build_multimodal_attention_mask(
                image_attention_mask=image_attention_mask,
                text_attention_mask=text_attention_mask,
                use_cls_token=self.use_cls_token,
            )
        )

        boundaries = calculate_sequence_boundaries(
            number_of_image_tokens=number_of_image_tokens,
            number_of_text_tokens=number_of_text_tokens,
            use_cls_token=self.use_cls_token,
        )

        return MultimodalSequenceOutput(
            embeddings=multimodal_embeddings,
            attention_mask=multimodal_attention_mask,
            multimodal_cls_index=boundaries[
                "multimodal_cls_index"
            ],
            image_token_start=boundaries[
                "image_token_start"
            ],
            image_token_end=boundaries[
                "image_token_end"
            ],
            text_token_start=boundaries[
                "text_token_start"
            ],
            text_token_end=boundaries[
                "text_token_end"
            ],
        )

    def _add_modality_embedding(
        self,
        *,
        embeddings: Tensor,
        modality_id: int,
    ) -> Tensor:
        if self.modality_embeddings is None:
            return embeddings

        modality_ids = torch.full(
            embeddings.shape[:2],
            fill_value=modality_id,
            dtype=torch.long,
            device=embeddings.device,
        )

        modality_embeddings = self.modality_embeddings(
            modality_ids
        )

        return embeddings + modality_embeddings

    def _validate_inputs(
        self,
        *,
        image_embeddings: Tensor,
        image_attention_mask: Tensor,
        text_embeddings: Tensor,
        text_attention_mask: Tensor,
    ) -> None:
        self._validate_embeddings(
            embeddings=image_embeddings,
            name="image_embeddings",
        )

        self._validate_embeddings(
            embeddings=text_embeddings,
            name="text_embeddings",
        )

        self._validate_mask(
            attention_mask=image_attention_mask,
            embeddings=image_embeddings,
            name="image_attention_mask",
        )

        self._validate_mask(
            attention_mask=text_attention_mask,
            embeddings=text_embeddings,
            name="text_attention_mask",
        )

        if (
            image_embeddings.shape[0]
            != text_embeddings.shape[0]
        ):
            raise ValueError(
                "Image and text embeddings must have the "
                "same batch size."
            )

        if image_embeddings.device != text_embeddings.device:
            raise ValueError(
                "Image and text embeddings must be on the "
                "same device."
            )

        if image_embeddings.dtype != text_embeddings.dtype:
            raise TypeError(
                "Image and text embeddings must use the "
                "same dtype."
            )

    def _validate_embeddings(
        self,
        *,
        embeddings: Tensor,
        name: str,
    ) -> None:
        if embeddings.ndim != 3:
            raise ValueError(
                f"{name} must have shape "
                "[batch_size, sequence_length, hidden_size]."
            )

        if embeddings.shape[0] <= 0:
            raise ValueError(
                f"{name} must contain at least one sample."
            )

        if embeddings.shape[1] <= 0:
            raise ValueError(
                f"{name} must contain at least one token."
            )

        if embeddings.shape[2] != self.hidden_size:
            raise ValueError(
                f"{name} hidden size must be "
                f"{self.hidden_size}, but received "
                f"{embeddings.shape[2]}."
            )

        if not embeddings.is_floating_point():
            raise TypeError(
                f"{name} must use a floating-point dtype."
            )

    def _validate_mask(
        self,
        *,
        attention_mask: Tensor,
        embeddings: Tensor,
        name: str,
    ) -> None:
        if attention_mask.ndim != 2:
            raise ValueError(
                f"{name} must have shape "
                "[batch_size, sequence_length]."
            )

        if attention_mask.shape != embeddings.shape[:2]:
            raise ValueError(
                f"{name} must match the batch and sequence "
                "dimensions of its embeddings."
            )

        if attention_mask.device != embeddings.device:
            raise ValueError(
                f"{name} and its embeddings must be on "
                "the same device."
            )