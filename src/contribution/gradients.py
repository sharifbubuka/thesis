from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class CapturedActivations:
    text: list[torch.Tensor]
    image: list[torch.Tensor]


class ViltActivationCapture(AbstractContextManager["ViltActivationCapture"]):
    """Capture differentiable text-embedding and image-patch activations.

    NLVR2 may invoke the shared ViLT encoder once per image; therefore activations
    are stored as lists and combined after backpropagation.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.values = CapturedActivations(text=[], image=[])
        self._handles: list[Any] = []

    def _capture_text(self, _module: torch.nn.Module, _inputs: Any, output: Any) -> None:
        tensor = output[0] if isinstance(output, tuple) else output
        if isinstance(tensor, torch.Tensor):
            tensor.retain_grad()
            self.values.text.append(tensor)

    def _capture_image(self, _module: torch.nn.Module, _inputs: Any, output: Any) -> None:
        tensor = output[0] if isinstance(output, tuple) else output
        if isinstance(tensor, torch.Tensor):
            tensor.retain_grad()
            self.values.image.append(tensor)

    def __enter__(self) -> "ViltActivationCapture":
        backbone = getattr(self.model, "vilt", None)
        if backbone is None:
            raise AttributeError("Expected a ViLT task model with a `.vilt` backbone.")

        embeddings = backbone.embeddings
        self._handles.append(embeddings.text_embeddings.register_forward_hook(self._capture_text))
        # Capturing the complete visual embedding sequence is more stable across
        # Transformers versions than depending on the internal Conv2d layout.
        self._handles.append(embeddings.patch_embeddings.register_forward_hook(self._capture_image))
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _flatten_token_axis(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor
    if tensor.ndim == 3:
        return tensor.reshape(-1, tensor.shape[-1])
    if tensor.ndim == 4:  # convolutional feature map: B,C,H,W
        return tensor.permute(0, 2, 3, 1).reshape(-1, tensor.shape[1])
    return tensor.reshape(-1, tensor.shape[-1])


def attribution_scores(activations: list[torch.Tensor]) -> tuple[float, float, torch.Tensor]:
    """Return gradient norm, |gradient × activation| total, and per-token scores."""
    if not activations:
        return 0.0, 0.0, torch.empty(0)

    token_scores: list[torch.Tensor] = []
    gradient_squared_sum = torch.tensor(0.0, device=activations[0].device)
    gx_total = torch.tensor(0.0, device=activations[0].device)

    for activation in activations:
        gradient = activation.grad
        if gradient is None:
            continue
        flat_activation = _flatten_token_axis(activation)
        flat_gradient = _flatten_token_axis(gradient)
        scores = (flat_activation * flat_gradient).abs().sum(dim=-1)
        token_scores.append(scores)
        gradient_squared_sum = gradient_squared_sum + gradient.pow(2).sum()
        gx_total = gx_total + scores.sum()

    if not token_scores:
        return 0.0, 0.0, torch.empty(0)

    return (
        float(gradient_squared_sum.sqrt().detach().cpu().item()),
        float(gx_total.detach().cpu().item()),
        torch.cat(token_scores).detach().cpu(),
    )
