"""Implement binary evaluation metrics in M6."""
from pathlib import Path
import torch


def binary_metrics(y_true: torch.Tensor, probabilities: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    """Input: targets/probabilities [N]. Output: metric names and scalar values.

    TODO: accuracy, precision, recall, F1, confusion matrix; zero division policy.
    """
    raise NotImplementedError("M6")
