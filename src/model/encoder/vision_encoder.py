from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from transformers import AutoConfig, AutoModel
from transformers.modeling_outputs import BaseModelOutputWithPooling

from ..contracts import VisionEncoderOutput

class VisionEncoder(nn.Module):
    """Pretrained ViT wrapper for extracting image patch embeddings."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()

        self.config = config
        self.vision_config = config["vision"]

        self.model_name: str = self.vision_config["model_name"]
        self.image_size: int = self.vision_config["image_size"]
        self.patch_size: int = self.vision_config["patch_size"]
        self.remove_cls_token: bool = self.vision_config["remove_cls_token"]

        self.output_hidden_states: bool = self.vision_config[
            "output_hidden_states"
        ]
        self.output_attentions: bool = self.vision_config[
            "output_attentions"
        ]

        self._validate_config()

        self.encoder = self._load_encoder()

        self.hidden_size = int(self.encoder.config.hidden_size)
        self.number_of_layers = int(
            self.encoder.config.num_hidden_layers
        )
        self.number_of_attention_heads = int(
            self.encoder.config.num_attention_heads
        )

        self._validate_loaded_encoder()
        self._apply_training_mode()

    def _validate_config(self) -> None:
        required_fields = {
            "model_name",
            "pretrained",
            "image_size",
            "patch_size",
            "remove_cls_token",
            "training_mode",
            "trainable_last_n_layers",
            "output_hidden_states",
            "output_attentions",
        }

        missing_fields = required_fields - self.vision_config.keys()

        if missing_fields:
            raise ValueError(
                "Missing vision configuration fields: "
                f"{sorted(missing_fields)}"
            )

        if not self.model_name:
            raise ValueError(
                "The vision model name cannot be empty."
            )

        if self.image_size <= 0:
            raise ValueError(
                "The vision image size must be positive."
            )

        if self.patch_size <= 0:
            raise ValueError(
                "The vision patch size must be positive."
            )

        if self.image_size % self.patch_size != 0:
            raise ValueError(
                "The image size must be divisible by the patch size."
            )

        valid_training_modes = {
            "frozen",
            "last_n_layers",
            "full",
        }

        training_mode = self.vision_config["training_mode"]

        if training_mode not in valid_training_modes:
            raise ValueError(
                f"Unsupported vision training mode '{training_mode}'. "
                f"Expected one of {sorted(valid_training_modes)}."
            )

        trainable_layers = self.vision_config[
            "trainable_last_n_layers"
        ]

        if trainable_layers < 0:
            raise ValueError(
                "trainable_last_n_layers cannot be negative."
            )

        if (
            training_mode == "last_n_layers"
            and trainable_layers == 0
        ):
            raise ValueError(
                "trainable_last_n_layers must be greater than zero "
                "when training_mode is 'last_n_layers'."
            )

    def _load_encoder(self) -> nn.Module:
        """
        Load a pretrained ViT or initialize one from its configuration.

        Eager attention is used when attention tensors are requested.
        """

        load_kwargs: dict[str, Any] = {
            "output_hidden_states": self.output_hidden_states,
            "output_attentions": self.output_attentions,
        }

        if self.output_attentions:
            load_kwargs["attn_implementation"] = "eager"

        if self.vision_config["pretrained"]:
            return AutoModel.from_pretrained(
                self.model_name,
                **load_kwargs,
            )

        huggingface_config = AutoConfig.from_pretrained(
            self.model_name,
            **load_kwargs,
        )

        return AutoModel.from_config(
            huggingface_config,
            attn_implementation=(
                "eager" if self.output_attentions else None
            ),
        )

    def _validate_loaded_encoder(self) -> None:
        loaded_image_size = int(self.encoder.config.image_size)
        loaded_patch_size = int(self.encoder.config.patch_size)

        if loaded_image_size != self.image_size:
            raise ValueError(
                "Configured image size does not match the loaded model: "
                f"configured={self.image_size}, "
                f"loaded={loaded_image_size}."
            )

        if loaded_patch_size != self.patch_size:
            raise ValueError(
                "Configured patch size does not match the loaded model: "
                f"configured={self.patch_size}, "
                f"loaded={loaded_patch_size}."
            )

        encoder_layers = self._get_encoder_layers()

        if len(encoder_layers) != self.number_of_layers:
            raise RuntimeError(
                "The number of loaded encoder layers is inconsistent "
                "with the Hugging Face model configuration."
            )

    def _get_encoder_layers(self) -> nn.ModuleList:
        """Return ViT blocks across supported Transformers layouts."""

        # Transformers 5 exposes ViT blocks directly as ``model.layers``.
        layers = getattr(self.encoder, "layers", None)

        if isinstance(layers, nn.ModuleList):
            return layers

        # Transformers 4 uses ``model.encoder.layer``.
        encoder = getattr(self.encoder, "encoder", None)
        layers = getattr(encoder, "layer", None)

        if isinstance(layers, nn.ModuleList):
            return layers

        raise TypeError(
            "The loaded vision model does not expose transformer layers "
            "through either layers or encoder.layer."
        )

    @property
    def patch_grid_size(self) -> tuple[int, int]:
        side = self.image_size // self.patch_size
        return side, side

    @property
    def number_of_patches(self) -> int:
        height, width = self.patch_grid_size
        return height * width

    def _set_all_parameters_trainable(
        self,
        trainable: bool,
    ) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = trainable

    def _apply_training_mode(self) -> None:
        training_mode = self.vision_config["training_mode"]
        trainable_last_n_layers = self.vision_config[
            "trainable_last_n_layers"
        ]

        if training_mode == "frozen":
            self._set_all_parameters_trainable(False)
            return

        if training_mode == "full":
            self._set_all_parameters_trainable(True)
            return

        self._set_all_parameters_trainable(False)

        if trainable_last_n_layers > self.number_of_layers:
            raise ValueError(
                "trainable_last_n_layers cannot exceed the number "
                f"of ViT encoder layers ({self.number_of_layers})."
            )

        trainable_layers = self._get_encoder_layers()[
            -trainable_last_n_layers:
        ]

        for layer in trainable_layers:
            for parameter in layer.parameters():
                parameter.requires_grad = True

        # The final layer normalization should adapt together with
        # the final trainable encoder layers.
        if hasattr(self.encoder, "layernorm"):
            for parameter in self.encoder.layernorm.parameters():
                parameter.requires_grad = True

    def train(self, mode: bool = True) -> VisionEncoder:
        """
        Set module training mode.

        A fully frozen encoder remains in evaluation mode so dropout
        and other training-only behaviour stay disabled.
        """

        super().train(mode)

        if self.vision_config["training_mode"] == "frozen":
            self.encoder.eval()

        return self

    def count_parameters(self) -> dict[str, int]:
        total = sum(
            parameter.numel()
            for parameter in self.encoder.parameters()
        )

        trainable = sum(
            parameter.numel()
            for parameter in self.encoder.parameters()
            if parameter.requires_grad
        )

        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }

    def forward(
        self,
        pixel_values: Tensor,
    ) -> VisionEncoderOutput:
        """
        Encode a batch of preprocessed images.

        Args:
            pixel_values:
                Float tensor with shape
                [batch_size, 3, image_size, image_size].

        Returns:
            Patch-level visual representations and optional
            intermediate states and attention matrices.
        """

        self._validate_pixel_values(pixel_values)

        outputs: BaseModelOutputWithPooling = self.encoder(
            pixel_values=pixel_values,
            output_hidden_states=self.output_hidden_states,
            output_attentions=self.output_attentions,
            return_dict=True,
        )

        sequence_embeddings = outputs.last_hidden_state

        expected_sequence_length = self.number_of_patches + 1

        if sequence_embeddings.shape[1] != expected_sequence_length:
            raise RuntimeError(
                "Unexpected ViT output sequence length: "
                f"expected={expected_sequence_length}, "
                f"received={sequence_embeddings.shape[1]}."
            )

        cls_embedding = sequence_embeddings[:, 0]
        patch_embeddings = sequence_embeddings[:, 1:]

        if not self.remove_cls_token:
            patch_embeddings = sequence_embeddings

        number_of_output_tokens = patch_embeddings.shape[1]

        attention_mask = torch.ones(
            pixel_values.shape[0],
            number_of_output_tokens,
            dtype=torch.long,
            device=pixel_values.device,
        )

        return VisionEncoderOutput(
            patch_embeddings=patch_embeddings,
            cls_embedding=cls_embedding,
            attention_mask=attention_mask,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            patch_grid_size=self.patch_grid_size,
        )

    def _validate_pixel_values(
        self,
        pixel_values: Tensor,
    ) -> None:
        if pixel_values.ndim != 4:
            raise ValueError(
                "pixel_values must have shape "
                "[batch_size, channels, height, width], "
                f"but received {tuple(pixel_values.shape)}."
            )

        batch_size, channels, height, width = pixel_values.shape

        if batch_size <= 0:
            raise ValueError(
                "pixel_values must contain at least one image."
            )

        expected_channels = int(
            getattr(self.encoder.config, "num_channels", 3)
        )

        if channels != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} image channels, "
                f"but received {channels}."
            )

        if height != self.image_size or width != self.image_size:
            raise ValueError(
                "Images must match the configured ViT resolution: "
                f"expected=({self.image_size}, {self.image_size}), "
                f"received=({height}, {width})."
            )

        if not pixel_values.is_floating_point():
            raise TypeError(
                "pixel_values must be a floating-point tensor. "
                "Use the model's image processor before encoding."
            )
