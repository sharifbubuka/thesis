import torch

def print_runtime():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
        print("GPU memory allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 2), "GB")
        print("GPU memory reserved:", round(torch.cuda.memory_reserved(0) / 1024**3, 2), "GB")
    else:
        print("No GPU detected.")