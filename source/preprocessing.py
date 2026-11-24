"""Define deterministic RGB resizing and tensor conversion in M1."""
from pathlib import Path
import torch


def build_preprocessing(image_size: tuple[int, int] = (64, 64)):
    """Input: (height,width). Output: callable PIL image -> float tensor [3,H,W].

    TODO: RGB conversion, resize, scale to [0,1]; document normalization.
    """
    raise NotImplementedError("M1")
