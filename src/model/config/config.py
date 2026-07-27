from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class VisionConfig:
    model_name: str = "google/vit-base-patch16-224-in21k"
    pretrained: bool = True
    image_size: int = 224
    patch_size: int = 16
    remove_cls_token: bool = True
    training_mode: str = "frozen"
    trainable_last_n_layers: int = 0
    output_hidden_states: bool = False
    output_attentions: bool = False


@dataclass(frozen=True, slots=True)
class TextConfig:
    model_name: str = "jhu-clsp/ettin-encoder-68m"
    tokenizer_name: str | None = None
    pretrained: bool = True
    max_length: int = 40
    training_mode: str = "frozen"
    trainable_last_n_layers: int = 0
    output_hidden_states: bool = False
    output_attentions: bool = False
    trust_remote_code: bool = False
    use_fast_tokenizer: bool = True


@dataclass(frozen=True, slots=True)
class ProjectionConfig:
    hidden_size: int = 384
    dropout: float = 0.1
    activation: str = "gelu"
    use_layer_norm: bool = True
    use_bias: bool = True


@dataclass(frozen=True, slots=True)
class MultimodalSequenceConfig:
    use_cls_token: bool = True
    use_modality_embeddings: bool = True
    dropout: float = 0.1
    use_layer_norm: bool = True
    number_of_modality_types: int = 3
    initialization_std: float = 0.02


@dataclass(frozen=True, slots=True)
class FusionConfig:
    hidden_size: int = 384
    number_of_layers: int = 4
    number_of_attention_heads: int = 6
    feed_forward_size: int = 1536
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation: str = "gelu"
    layer_norm_epsilon: float = 1e-5
    use_final_layer_norm: bool = True


@dataclass(frozen=True, slots=True)
class PoolingConfig:
    strategy: str = "cls"
    use_layer_norm: bool = True
    dropout: float = 0.1


@dataclass(frozen=True, slots=True)
class ClassifierConfig:
    number_of_classes: int = 3_129
    hidden_size: int = 768
    dropout: float = 0.1
    activation: str = "gelu"
    use_layer_norm: bool = True
    use_bias: bool = True


@dataclass(frozen=True, slots=True)
class CompactViltConfig:
    model_name: str = "compact-vilt-ettin"
    initialization_seed: int = 42
    vision: VisionConfig = field(default_factory=VisionConfig)
    text: TextConfig = field(default_factory=TextConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    multimodal_sequence: MultimodalSequenceConfig = field(
        default_factory=MultimodalSequenceConfig
    )
    fusion: FusionConfig = field(default_factory=FusionConfig)
    pooling: PoolingConfig = field(default_factory=PoolingConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)

    def with_number_of_classes(self, number_of_classes: int) -> "CompactViltConfig":
        return CompactViltConfig(
            model_name=self.model_name,
            initialization_seed=self.initialization_seed,
            vision=self.vision,
            text=self.text,
            projection=self.projection,
            multimodal_sequence=self.multimodal_sequence,
            fusion=self.fusion,
            pooling=self.pooling,
            classifier=ClassifierConfig(
                number_of_classes=number_of_classes,
                hidden_size=self.classifier.hidden_size,
                dropout=self.classifier.dropout,
                activation=self.classifier.activation,
                use_layer_norm=self.classifier.use_layer_norm,
                use_bias=self.classifier.use_bias,
            ),
        )

    def with_analysis_outputs(self, enabled: bool = True) -> "CompactViltConfig":
        vision = VisionConfig(**{**asdict(self.vision), "output_hidden_states": enabled, "output_attentions": enabled})
        text = TextConfig(**{**asdict(self.text), "output_hidden_states": enabled, "output_attentions": enabled})
        return CompactViltConfig(
            model_name=self.model_name,
            initialization_seed=self.initialization_seed,
            vision=vision,
            text=text,
            projection=self.projection,
            multimodal_sequence=self.multimodal_sequence,
            fusion=self.fusion,
            pooling=self.pooling,
            classifier=self.classifier,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelConfig:
    @staticmethod
    def compact_vilt() -> CompactViltConfig:
        return CompactViltConfig()
