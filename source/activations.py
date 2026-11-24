"""Implement custom activations and derivatives in M3."""
from pathlib import Path
import torch


def activation_forward(x: torch.Tensor, name: str) -> torch.Tensor:
    """Input/output: tensors of equal shape. TODO: implement and validate formula."""
    raise NotImplementedError("M3")


def activation_derivative(x: torch.Tensor, name: str) -> torch.Tensor:
    """Input/output: tensors of equal shape. TODO: check numerical derivatives."""
    raise NotImplementedError("M3")
