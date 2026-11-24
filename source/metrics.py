import math
import torch


def binary_metrics(
    y_true: torch.Tensor,
    probabilities: torch.Tensor,
    threshold: float = 0.5
) -> dict[str, float]:

    """
    Calculate binary classification metrics.

    Positive class: Human (1).
    Negative class: Horse (0).

    Zero division policy: return 0.0.
    """

    # 1. Same shape
    if probabilities.shape != y_true.shape:
        raise ValueError(
            "Probabilities and y_true must have the same shape."
        )

    # 2. One-dimensional tensors
    if probabilities.ndim != 1 or y_true.ndim != 1:
        raise ValueError(
            "Probabilities and y_true must be one-dimensional tensors."
        )

    # 3. Non-empty tensors
    if probabilities.numel() == 0 or y_true.numel() == 0:
        raise ValueError(
            "Probabilities and y_true must not be empty."
        )

    # 4. Finite probabilities
    if not torch.isfinite(probabilities).all():
        raise ValueError(
            "Probabilities must contain only finite values."
        )

    # 5. Probabilities in [0, 1]
    if not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError(
            "Probabilities must be in [0, 1]."
        )

    # 6. Targets are 0 or 1
    if not ((y_true == 0) | (y_true == 1)).all():
        raise ValueError(
            "y_true must contain only 0 or 1."
        )

    # 7. Threshold in [0, 1]
    if not (0 <= threshold <= 1):
        raise ValueError(
            "Threshold must be in [0, 1]."
        )

    predictions = (probabilities >= threshold).long()

    tp = ((y_true == 1) & (predictions == 1)).sum().item()
    tn = ((y_true == 0) & (predictions == 0)).sum().item()
    fp = ((y_true == 0) & (predictions == 1)).sum().item()
    fn = ((y_true == 1) & (predictions == 0)).sum().item()


    accuracy = (tp + tn) / len(y_true)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    recall = tp/(tp + fn) if (tp + fn) > 0 else 0.0

    f1 = (2 * tp)/(2* tp + fp + fn) if (2* tp + fp + fn) > 0 else 0.0


    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn)
    }
