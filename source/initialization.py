"""Implement weight initialization methods in M4."""
from pathlib import Path
import torch
import math


def initialize_weights(
    module: torch.nn.Module,
    method: str = "he_normal"
) -> None:
    """
    Initialize weights of an nn.Linear layer.

    Supported methods:
        - normal
        - fan_in_normal
        - xavier_normal
        - he_normal

    Parameters are initialized in-place.
    """

    supported = {"normal", "fan_in_normal", "xavier_normal", "he_normal"}

    if method not in supported:
        raise ValueError(f"Unsupported initialization method: {method}")

    if not isinstance(module, torch.nn.Linear):
        return


    fan_in = module.in_features
    fan_out = module.out_features

    # Calculate standard deviation for each method
    std = None
    if method == "normal":
        std = 1.0

    elif method == "fan_in_normal":
        std = math.sqrt(1/fan_in)

    elif method == "xavier_normal":
        std = math.sqrt(2/(fan_in + fan_out))

    elif method == "he_normal":
        std = math.sqrt(2/fan_in)

    with torch.no_grad():

        module.weight.normal_(mean=0.0, std=std)

        if module.bias is not None:
            module.bias.zero_()