import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

class LlavaModelLoader:
    """
    Loads LLaVA model and processor with optional 4-bit quantization.
    """

    def __init__(self, config: dict):
        self.config = config
        self.model_name = config["base_model_name"]

    def get_quantization_config(self):
        """
        Create 4-bit quantization config for memory-efficient loading.
        """
        if not self.config.get("use_4bit", True):
            return None

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    def load_processor(self):
        """
        Load processor for image-text inputs.
        """
        return AutoProcessor.from_pretrained(self.model_name)

    def load_model(self):
        """
        Load LLaVA model.
        """
        quantization_config = self.get_quantization_config()

        model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        model.eval()
        return model

    def load(self):
        """
        Load both model and processor.
        """
        processor = self.load_processor()
        model = self.load_model()

        return model, processor