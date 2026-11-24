"""Implement weight initialization methods in M4."""
from pathlib import Path
import torch


def initialize_weights(module: torch.nn.Module, method: str = "he_normal") -> None:
    """Input: module/method. Output: in-place initialized parameters.

    TODO: fan-in/fan-out scaling, bias initialization, seeded reproducibility.
    """
    raise NotImplementedError("M4")
