from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput

from ..contracts import TextEncoderOutput


class TextEncoder(nn.Module):
    """Ettin wrapper for tokenization and contextual text encoding."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()

        self.config = config
        self.text_config = config["text"]

        self.model_name: str = self.text_config["model_name"]
        self.tokenizer_name: str = (
            self.text_config["tokenizer_name"]
            or self.model_name
        )
        self.max_length: int = self.text_config["max_length"]

        self.output_hidden_states: bool = self.text_config[
            "output_hidden_states"
        ]
        self.output_attentions: bool = self.text_config[
            "output_attentions"
        ]

        self._validate_config()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            use_fast=self.text_config["use_fast_tokenizer"],
            trust_remote_code=self.text_config[
                "trust_remote_code"
            ],
        )

        self.encoder = self._load_encoder()

        self.hidden_size = int(self.encoder.config.hidden_size)
        self.number_of_layers = int(
            self.encoder.config.num_hidden_layers
        )
        self.number_of_attention_heads = int(
            self.encoder.config.num_attention_heads
        )

        self.encoder_layers = self._resolve_encoder_layers()

        self._apply_training_mode()

    def _validate_config(self) -> None:
        required_fields = {
            "model_name",
            "tokenizer_name",
            "pretrained",
            "max_length",
            "training_mode",
            "trainable_last_n_layers",
            "output_hidden_states",
            "output_attentions",
            "trust_remote_code",
            "use_fast_tokenizer",
        }

        missing_fields = required_fields - self.text_config.keys()

        if missing_fields:
            raise ValueError(
                "Missing text configuration fields: "
                f"{sorted(missing_fields)}"
            )

        if not self.model_name:
            raise ValueError(
                "The text model name cannot be empty."
            )

        if self.max_length <= 0:
            raise ValueError(
                "The text maximum length must be positive."
            )

        training_mode = self.text_config["training_mode"]

        valid_modes = {
            "frozen",
            "last_n_layers",
            "full",
        }

        if training_mode not in valid_modes:
            raise ValueError(
                f"Unsupported text training mode '{training_mode}'. "
                f"Expected one of {sorted(valid_modes)}."
            )

        trainable_layers = self.text_config[
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
        load_kwargs: dict[str, Any] = {
            "output_hidden_states": self.output_hidden_states,
            "output_attentions": self.output_attentions,
            "trust_remote_code": self.text_config[
                "trust_remote_code"
            ],
        }

        if self.output_attentions:
            load_kwargs["attn_implementation"] = "eager"

        if self.text_config["pretrained"]:
            return AutoModel.from_pretrained(
                self.model_name,
                **load_kwargs,
            )

        model_config = AutoConfig.from_pretrained(
            self.model_name,
            **load_kwargs,
        )

        return AutoModel.from_config(
            model_config,
            trust_remote_code=self.text_config[
                "trust_remote_code"
            ],
        )

    def _resolve_encoder_layers(self) -> nn.ModuleList:
        """
        Find the Transformer layers without depending on one exact
        Hugging Face architecture name.
        """

        candidate_paths = (
            # ModernBERT (and therefore Ettin) stores its blocks
            # directly on the base model.
            ("layers",),
            ("encoder", "layer"),
            ("encoder", "layers"),
            ("transformer", "layer"),
            ("transformer", "layers"),
        )

        base_model = getattr(
            self.encoder,
            "base_model",
            self.encoder,
        )

        for path in candidate_paths:
            layers: Any = base_model

            for name in path:
                layers = getattr(layers, name, None)

                if layers is None:
                    break

            if isinstance(layers, nn.ModuleList):
                if len(layers) != self.number_of_layers:
                    raise RuntimeError(
                        "Resolved text encoder layers do not match "
                        "the configured number of layers."
                    )

                return layers

        raise TypeError(
            "Could not locate the Transformer layers in the "
            "loaded text encoder."
        )

    def _set_all_parameters_trainable(
        self,
        trainable: bool,
    ) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = trainable

    def _apply_training_mode(self) -> None:
        training_mode = self.text_config["training_mode"]
        last_n_layers = self.text_config[
            "trainable_last_n_layers"
        ]

        if training_mode == "frozen":
            self._set_all_parameters_trainable(False)
            return

        if training_mode == "full":
            self._set_all_parameters_trainable(True)
            return

        if last_n_layers > self.number_of_layers:
            raise ValueError(
                "trainable_last_n_layers cannot exceed the number "
                f"of text encoder layers ({self.number_of_layers})."
            )

        self._set_all_parameters_trainable(False)

        for layer in self.encoder_layers[-last_n_layers:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

        self._unfreeze_final_normalization()

    def _unfreeze_final_normalization(self) -> None:
        """
        Unfreeze a final normalization layer when the architecture
        exposes one.
        """

        base_model = getattr(
            self.encoder,
            "base_model",
            self.encoder,
        )

        candidate_names = (
            "layernorm",
            "layer_norm",
            "final_norm",
            "final_layer_norm",
        )

        for name in candidate_names:
            normalization = getattr(
                base_model,
                name,
                None,
            )

            if isinstance(normalization, nn.Module):
                for parameter in normalization.parameters():
                    parameter.requires_grad = True

                return

    def train(self, mode: bool = True) -> TextEncoder:
        super().train(mode)

        if self.text_config["training_mode"] == "frozen":
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

    def tokenize(
        self,
        texts: str | Sequence[str],
        *,
        device: torch.device | str | None = None,
    ) -> dict[str, Tensor]:
        """
        Tokenize one question or a batch of questions.
        """

        normalized_texts = self._normalize_texts(texts)

        encoded = self.tokenizer(
            normalized_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        if device is not None:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def forward(
        self,
        *,
        texts: str | Sequence[str] | None = None,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        return_tokens: bool = False,
    ) -> TextEncoderOutput:
        """
        Encode either raw text or already-tokenized inputs.

        Exactly one input style should be used:

            text_encoder(texts=["...", "..."])

        or:

            text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        """

        input_ids, attention_mask = self._resolve_inputs(
            texts=texts,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        outputs: BaseModelOutput = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=self.output_hidden_states,
            output_attentions=self.output_attentions,
            return_dict=True,
        )

        token_embeddings = outputs.last_hidden_state

        if token_embeddings.ndim != 3:
            raise RuntimeError(
                "The text encoder output must have shape "
                "[batch_size, sequence_length, hidden_size]."
            )

        if token_embeddings.shape[:2] != input_ids.shape:
            raise RuntimeError(
                "The text encoder output sequence dimensions do "
                "not match the input IDs."
            )

        tokens = None

        if return_tokens:
            tokens = self._convert_ids_to_tokens(input_ids)

        return TextEncoderOutput(
            token_embeddings=token_embeddings,
            attention_mask=attention_mask,
            input_ids=input_ids,
            tokens=tokens,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _resolve_inputs(
        self,
        *,
        texts: str | Sequence[str] | None,
        input_ids: Tensor | None,
        attention_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        using_texts = texts is not None
        using_ids = input_ids is not None

        if using_texts == using_ids:
            raise ValueError(
                "Provide either texts or input_ids, but not both."
            )

        if texts is not None:
            device = next(self.encoder.parameters()).device

            encoded = self.tokenize(
                texts,
                device=device,
            )

            return (
                encoded["input_ids"],
                encoded["attention_mask"],
            )

        if input_ids is None:
            raise RuntimeError(
                "input_ids unexpectedly resolved to None."
            )

        self._validate_input_ids(input_ids)

        if attention_mask is None:
            attention_mask = self._build_attention_mask(
                input_ids
            )
        else:
            self._validate_attention_mask(
                input_ids,
                attention_mask,
            )

        encoder_device = next(
            self.encoder.parameters()
        ).device

        input_ids = input_ids.to(encoder_device)
        attention_mask = attention_mask.to(
            device=encoder_device,
            dtype=torch.long,
        )

        return input_ids, attention_mask

    def _build_attention_mask(
        self,
        input_ids: Tensor,
    ) -> Tensor:
        pad_token_id = self.tokenizer.pad_token_id

        if pad_token_id is None:
            return torch.ones_like(
                input_ids,
                dtype=torch.long,
            )

        return input_ids.ne(pad_token_id).long()

    def _normalize_texts(
        self,
        texts: str | Sequence[str],
    ) -> list[str]:
        if isinstance(texts, str):
            normalized = [texts]
        else:
            normalized = list(texts)

        if not normalized:
            raise ValueError(
                "At least one text input is required."
            )

        for text in normalized:
            if not isinstance(text, str):
                raise TypeError(
                    "Every text input must be a string."
                )

            if not text.strip():
                raise ValueError(
                    "Text inputs cannot be empty."
                )

        return normalized

    def _validate_input_ids(
        self,
        input_ids: Tensor,
    ) -> None:
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape "
                "[batch_size, sequence_length]."
            )

        if input_ids.shape[0] <= 0:
            raise ValueError(
                "input_ids must contain at least one sample."
            )

        if input_ids.shape[1] <= 0:
            raise ValueError(
                "input_ids must contain at least one token."
            )

        if input_ids.shape[1] > self.max_length:
            raise ValueError(
                f"Input sequence length {input_ids.shape[1]} "
                f"exceeds max_length={self.max_length}."
            )

        if input_ids.dtype not in {
            torch.int32,
            torch.int64,
        }:
            raise TypeError(
                "input_ids must use an integer dtype."
            )

    def _validate_attention_mask(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> None:
        if attention_mask.ndim != 2:
            raise ValueError(
                "attention_mask must have shape "
                "[batch_size, sequence_length]."
            )

        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                "attention_mask must have the same shape as "
                "input_ids."
            )

    def _convert_ids_to_tokens(
        self,
        input_ids: Tensor,
    ) -> tuple[tuple[str, ...], ...]:
        rows = input_ids.detach().cpu().tolist()

        converted = []

        for row in rows:
            tokens = self.tokenizer.convert_ids_to_tokens(row)
            converted.append(tuple(tokens))

        return tuple(converted)
