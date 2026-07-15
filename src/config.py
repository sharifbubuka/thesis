from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_ROOT = PROJECT_ROOT.parent / "src" / "outputs"

CONFIG = {
    "seed": 42,
    "data": {
        "datasets": ["textvqa", "gqa", "vqav2"],
        "max_size": 1_000,
    },
    "model": {
        "models": ["llava-hf/llava-1.5-7b-hf"],
        "active_model": "llava-hf/llava-1.5-7b-hf",
        "use_4bit": True,
    },

    # Model settings
    # "base_model_name": "llava-hf/llava-1.5-7b-hf",
    # "use_4bit": True,
    # "training_method": "qlora",  # options: "lora", "qlora"

    # # Inference settings
    # "max_new_tokens": 32,
    # "batch_size": 1,

    # Masking settings
    "mask_strategies": {
        "image": ["black_image"],
        "text": ["neutral_prompt"],
    },
    "directories": {
        "datasets": PROJECT_ROOT / "src/data/datasets",
        "models": PROJECT_ROOT / "src/model/models",
        "results": OUTPUT_ROOT / "results",
        "figures": OUTPUT_ROOT / "figures",
        "checkpoints": OUTPUT_ROOT / "checkpoints",
        "logs": OUTPUT_ROOT / "logs",
    },
}
