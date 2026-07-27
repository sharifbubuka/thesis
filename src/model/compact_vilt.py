from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from torch import Tensor, nn

from .classifier import VQAClassificationHead
from .contracts import CompactViltOutput
from .encoder import TextEncoder, VisionEncoder
from .fusion import MultimodalFusionTransformer
from .multimodal import MultimodalSequenceBuilder
from .pooling import MultimodalPooler
from .projection import ModalityProjection


class CompactViltModel(nn.Module):
    """
    Compact multimodal VQA model.

    Processing pipeline:

        image
          -> VisionEncoder
          -> image projection

        question
          -> TextEncoder
          -> text projection

        projected modalities
          -> MultimodalSequenceBuilder
          -> MultimodalFusionTransformer
          -> MultimodalPooler
          -> VQAClassificationHead
    """

    def __init__(
        self,
        config: dict[str, Any],
    ) -> None:
        super().__init__()

        self.config = config

        self.vision_encoder = VisionEncoder(
            config
        )

        self.text_encoder = TextEncoder(
            config
        )

        self.image_projection = ModalityProjection(
            input_size=self.vision_encoder.hidden_size,
            config=config,
            modality_name="image",
        )

        self.text_projection = ModalityProjection(
            input_size=self.text_encoder.hidden_size,
            config=config,
            modality_name="text",
        )

        self.sequence_builder = MultimodalSequenceBuilder(
            config
        )

        self.fusion_transformer = (
            MultimodalFusionTransformer(
                config
            )
        )

        self.pooler = MultimodalPooler(
            config
        )

        self.classifier = VQAClassificationHead(
            config
        )

        self._validate_model_configuration()

    def _validate_model_configuration(self) -> None:
        projection_hidden_size = self.config[
            "projection"
        ]["hidden_size"]

        fusion_hidden_size = self.config[
            "fusion"
        ]["hidden_size"]

        if projection_hidden_size != fusion_hidden_size:
            raise ValueError(
                "Projection and fusion hidden sizes must match: "
                f"projection={projection_hidden_size}, "
                f"fusion={fusion_hidden_size}."
            )

        pooling_strategy = self.config[
            "pooling"
        ]["strategy"]

        use_cls_token = self.config[
            "multimodal_sequence"
        ]["use_cls_token"]

        if pooling_strategy == "cls" and not use_cls_token:
            raise ValueError(
                "CLS pooling requires "
                "multimodal_sequence.use_cls_token=True."
            )

    def forward(
        self,
        *,
        pixel_values: Tensor,
        texts: str | Sequence[str] | None = None,
        input_ids: Tensor | None = None,
        text_attention_mask: Tensor | None = None,
        return_tokens: bool = False,
    ) -> CompactViltOutput:
        """
        Run the complete multimodal VQA pipeline.

        Provide text in exactly one of two forms:

        Raw questions:
            model(
                pixel_values=images,
                texts=questions,
            )

        Pretokenized questions:
            model(
                pixel_values=images,
                input_ids=input_ids,
                text_attention_mask=attention_mask,
            )
        """

        self._validate_forward_inputs(
            pixel_values=pixel_values,
            texts=texts,
            input_ids=input_ids,
            text_attention_mask=text_attention_mask,
        )

        vision_output = self.vision_encoder(
            pixel_values
        )

        text_output = self.text_encoder(
            texts=texts,
            input_ids=input_ids,
            attention_mask=text_attention_mask,
            return_tokens=return_tokens,
        )

        image_projection_output = self.image_projection(
            vision_output.patch_embeddings,
            vision_output.attention_mask,
        )

        text_projection_output = self.text_projection(
            text_output.token_embeddings,
            text_output.attention_mask,
        )

        sequence_output = self.sequence_builder(
            image_embeddings=(
                image_projection_output.embeddings
            ),
            image_attention_mask=(
                image_projection_output.attention_mask
            ),
            text_embeddings=(
                text_projection_output.embeddings
            ),
            text_attention_mask=(
                text_projection_output.attention_mask
            ),
        )

        fusion_output = self.fusion_transformer(
            embeddings=sequence_output.embeddings,
            attention_mask=sequence_output.attention_mask,
            multimodal_cls_index=(
                sequence_output.multimodal_cls_index
            ),
        )

        pooled_output = self.pooler(
            hidden_states=(
                fusion_output.last_hidden_state
            ),
            attention_mask=(
                fusion_output.attention_mask
            ),
            multimodal_cls_index=(
                sequence_output.multimodal_cls_index
            ),
        )

        classification_output = self.classifier(
            pooled_output
        )

        return CompactViltOutput(
            logits=classification_output.logits,
            scores=classification_output.scores,
            predicted_ids=(
                classification_output.predicted_ids
            ),
            confidence=classification_output.confidence,
            binary_entropy=(
                classification_output.binary_entropy
            ),
            image_patch_embeddings=(
                vision_output.patch_embeddings
            ),
            text_token_embeddings=(
                text_output.token_embeddings
            ),
            projected_image_embeddings=(
                image_projection_output.embeddings
            ),
            projected_text_embeddings=(
                text_projection_output.embeddings
            ),
            multimodal_embeddings=(
                sequence_output.embeddings
            ),
            fused_hidden_state=(
                fusion_output.last_hidden_state
            ),
            pooled_output=pooled_output,
            vision_hidden_states=(
                vision_output.hidden_states
            ),
            text_hidden_states=(
                text_output.hidden_states
            ),
            fusion_hidden_states=(
                fusion_output.hidden_states
            ),
            vision_attentions=(
                vision_output.attentions
            ),
            text_attentions=(
                text_output.attentions
            ),
            fusion_attentions=(
                fusion_output.attentions
            ),
            image_attention_mask=(
                image_projection_output.attention_mask
            ),
            text_attention_mask=(
                text_projection_output.attention_mask
            ),
            multimodal_attention_mask=(
                sequence_output.attention_mask
            ),
            multimodal_cls_index=(
                sequence_output.multimodal_cls_index
            ),
            image_token_start=(
                sequence_output.image_token_start
            ),
            image_token_end=(
                sequence_output.image_token_end
            ),
            text_token_start=(
                sequence_output.text_token_start
            ),
            text_token_end=(
                sequence_output.text_token_end
            ),
            patch_grid_size=(
                vision_output.patch_grid_size
            ),
            text_tokens=text_output.tokens,
        )

    def _validate_forward_inputs(
        self,
        *,
        pixel_values: Tensor,
        texts: str | Sequence[str] | None,
        input_ids: Tensor | None,
        text_attention_mask: Tensor | None,
    ) -> None:
        using_raw_text = texts is not None
        using_token_ids = input_ids is not None

        if using_raw_text == using_token_ids:
            raise ValueError(
                "Provide exactly one text input style: either "
                "texts or input_ids."
            )

        if input_ids is not None and text_attention_mask is None:
            raise ValueError(
                "text_attention_mask is required when input_ids "
                "are provided."
            )

        if texts is not None and text_attention_mask is not None:
            raise ValueError(
                "text_attention_mask must not be supplied when "
                "raw texts are provided."
            )

        if input_ids is not None:
            if input_ids.ndim != 2:
                raise ValueError(
                    "input_ids must have shape "
                    "[batch_size, sequence_length]."
                )

            if text_attention_mask is None:
                raise RuntimeError(
                    "Missing text attention mask after validation."
                )

            if text_attention_mask.shape != input_ids.shape:
                raise ValueError(
                    "text_attention_mask must have the same shape "
                    "as input_ids."
                )

            if input_ids.shape[0] != pixel_values.shape[0]:
                raise ValueError(
                    "Image and text batch sizes must match."
                )

        if texts is not None:
            number_of_texts = (
                1
                if isinstance(texts, str)
                else len(texts)
            )

            if number_of_texts != pixel_values.shape[0]:
                raise ValueError(
                    "The number of questions must match the "
                    "image batch size: "
                    f"images={pixel_values.shape[0]}, "
                    f"questions={number_of_texts}."
                )

    def count_parameters(
        self,
    ) -> dict[str, int]:
        """
        Count complete-model parameters.
        """

        total = sum(
            parameter.numel()
            for parameter in self.parameters()
        )

        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }