"""Implement optimization and validation loops in M6."""
from pathlib import Path
import torch


def train_epoch(model, loader, optimizer, criterion, device: str) -> dict:
    """Input: training dependencies. Output: epoch statistics.

    TODO: train mode, gradient reset, logits/loss, backward, optimizer step.
    """
    raise NotImplementedError("M6")


def evaluate(model, loader, criterion, device: str) -> dict:
    """Input: evaluation dependencies. Output: aggregate loss and metrics.

    TODO: eval mode, no_grad, sample-weighted aggregation.
    """
    raise NotImplementedError("M6")
