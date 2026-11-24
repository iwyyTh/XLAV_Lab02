"""Plot learning curves and model results in M8; EDA lives in its notebook."""
from pathlib import Path
import torch


def plot_history(history: dict, output_path: Path) -> None:
    """Input: epoch statistics and destination. Output: saved figure.

    TODO: plot train/validation loss and metrics with legends.
    """
    raise NotImplementedError("M8")
