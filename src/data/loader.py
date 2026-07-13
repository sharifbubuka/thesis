from datasets import load_dataset, disable_progress_bars
from datasets.utils.logging import set_verbosity_error
from huggingface_hub.utils import disable_progress_bars as hf_disable_progress_bars

from .registry import REGISTRY

set_verbosity_error()
disable_progress_bars()
hf_disable_progress_bars()

class HuggingFaceDatasetLoader:
    """
    Loads datasets from Hugging Face using the dataset registry.
    """

    def __init__(self, registry=None):
        self.registry = registry or REGISTRY

    def load(self, dataset_key: str):
        if dataset_key not in self.registry:
            raise ValueError(f"Unknown dataset key: {dataset_key}")

        info = self.registry[dataset_key]

        return load_dataset(
            info["hf_name"],
            split=info["split"],
        )