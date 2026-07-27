from __future__ import annotations

import torch


def resolve_device(preferred: str = "auto") -> torch.device:
    """Resolve the device used for model execution."""

    valid_options = {"auto", "cpu", "cuda", "mps"}

    if preferred not in valid_options:
        raise ValueError(
            f"Unsupported device '{preferred}'. "
            f"Expected one of {sorted(valid_options)}."
        )

    if preferred == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")

        return torch.device("cuda")

    if preferred == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")

        return torch.device("mps")

    if preferred == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")