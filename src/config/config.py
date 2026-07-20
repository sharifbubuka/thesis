from pathlib import Path
from ..utils.device import resolve_device

# config.py lives at <project>/src/config/config.py.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_ROOT = PROJECT_ROOT / "src" / "outputs"

global_config = {
    "seed": 42,
    "device": resolve_device("auto"),
    "data": {
        "datasets": ["textvqa", "gqa", "vqav2"],
        "max_size": 1_00,
    },
    "model": {
        "models": ["llava-hf/llava-1.5-7b-hf"],
        "active_model": "llava-hf/llava-1.5-7b-hf",
        "use_4bit": True,
    },

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
