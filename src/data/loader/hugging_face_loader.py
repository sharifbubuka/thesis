from pathlib import Path
from ...utils import print_message
from datasets import disable_progress_bars, load_dataset, load_from_disk
from datasets.utils.logging import set_verbosity_error
from huggingface_hub.utils import disable_progress_bars as hf_disable_progress_bars

from ..config import data_config as default_data_config

set_verbosity_error()
disable_progress_bars()
hf_disable_progress_bars()

class HuggingFaceDatasetLoader:
    """
    Loads datasets from Hugging Face using the dataset registry.
    """

    def __init__(self, registry: dict | None = None, config: dict | None = None):
        self.dataset = None
        self.config = config
        self.data_config = registry if registry is not None else default_data_config

    def load(self, dataset_key: str):
        if dataset_key not in self.data_config:
            raise ValueError(f"Unknown dataset key: {dataset_key}")
        
        dataset_path = Path(self.config["directories"]["datasets"]) / dataset_key
    
        if dataset_path.exists() and any(dataset_path.iterdir()):
            self.dataset = load_from_disk(str(dataset_path))
        else:
            dataset_info = self.data_config[dataset_key]

            self.dataset = load_dataset(
                dataset_info["hf_name"],
                split=dataset_info["split"],
            )
            
            self.dataset.save_to_disk(str(dataset_path))
        
        print_message(f"Loaded dataset {dataset_key} with {len(self.dataset)} samples", skip_line=False)
        
        return self.dataset
    
    def save(self, dataset_key: str):
        if self.dataset is None:
            raise ValueError("No dataset loaded to save.")
        dataset_path = Path(self.config["directories"]["datasets"]) / dataset_key
        if dataset_path.exists() and any(dataset_path.iterdir()):
            print(f"Dataset already exists at {dataset_path}. Skipping save.")
        else:
            self.dataset.save_to_disk(str(dataset_path))
            print(f"Dataset saved to {dataset_path}.")
