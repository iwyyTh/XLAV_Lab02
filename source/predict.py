"""Provide inference from a trained checkpoint in M6."""
from pathlib import Path
import torch


def predict(image_path: Path, checkpoint_path: Path, config) -> dict:
    """Input: image/checkpoint/config. Output: label and human probability.

    TODO: load checkpoint, eval preprocessing, no_grad, sigmoid and threshold.
    """
    raise NotImplementedError("M6")
