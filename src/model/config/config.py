config = {
    "model_name": "compact-vilt-ettin",
    "initialization_seed": 42,

    "vision": {
        "model_name": "google/vit-base-patch16-224-in21k",
        "pretrained": True,
        "image_size": 224,
        "patch_size": 16,
        "remove_cls_token": True,
        "training_mode": "frozen",
        "trainable_last_n_layers": 0,
        "output_hidden_states": True,
        "output_attentions": True,
    },

    "text": {
        "model_name": "jhu-clsp/ettin-encoder-68m",
        "tokenizer_name": None,
        "pretrained": True,
        "max_length": 40,
        "training_mode": "frozen",
        "trainable_last_n_layers": 0,
        "output_hidden_states": True,
        "output_attentions": True,
        "trust_remote_code": False,
        "use_fast_tokenizer": True,
    },

    "projection": {
        "hidden_size": 384,
        "dropout": 0.1,
        "activation": "gelu",
        "use_layer_norm": True,
        "use_bias": True,
    },

    "multimodal_sequence": {
        "use_cls_token": True,
        "use_modality_embeddings": True,
        "number_of_modality_types": 3,
        "initialization_std": 0.02,
    },

    "fusion": {
        "hidden_size": 384,
        "number_of_layers": 4,
        "number_of_attention_heads": 6,
        "intermediate_size": 1536,
        "dropout": 0.1,
        "attention_dropout": 0.1,
        "activation": "gelu",
        "norm_first": True,
        "layer_norm_epsilon": 1e-5,
        "output_hidden_states": True,
        "output_attentions": True,
    },

    "pooling": {
        "strategy": "cls",
        "use_layer_norm": True,
        "dropout": 0.1,
    },

    "classifier": {
        "number_of_classes": 1000,
        "hidden_size": 384,
        "intermediate_size": 768,
        "activation": "gelu",
        "dropout": 0.1,
        "use_layer_norm": True,
        "use_bias": True,
    },
}