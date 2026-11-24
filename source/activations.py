"""Implement custom activations and derivatives in M3."""
from pathlib import Path
import torch

def activation_forward(
    x: torch.Tensor,
    name: str
) -> torch.Tensor:
    """
    Apply a custom activation function.

    Supported:
        - relu
        - sigmoid

    Input and output have the same shape.
    """

    if name == "relu":

        result = torch.where(
            x > 0,
            x,
            torch.zeros_like(x)
        )

        return result

    elif name == "sigmoid":
        positive_mask = x >= 0
        negative_mask = x < 0

        result = torch.full_like(x, float("nan"))

        # Positive inputs
        result[positive_mask] = 1 / (
            1 + torch.exp(-x[positive_mask])
        )

        # Negative inputs
        exp_x = torch.exp(x[negative_mask])

        result[negative_mask] = exp_x / (1 + exp_x)

        return result

    else:
        raise ValueError(
            f"Unsupported activation: {name}"
        )


def activation_derivative(
    x: torch.Tensor,
    name: str
) -> torch.Tensor:
    """
    Calculate the derivative with respect to
    the activation input x.
    """

    if name == "relu":
        mask = x > 0
        return mask.to(dtype=x.dtype)

    elif name == "sigmoid":
        s = activation_forward(x, "sigmoid")
        return s * (1 - s)

    else:
        raise ValueError(
            f"Unsupported activation: {name}"
        )


class CustomActivationFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, name):

        # Save input tensor for backward
        ctx.save_for_backward(x)

        # Save activation name
        ctx.name = name

        # Apply custom forward
        return activation_forward(x, name)

    @staticmethod
    def backward(ctx, grad_output):

        # Retrieve saved information
        x, = ctx.saved_tensors
        name = ctx.name

        # Calculate local derivative
        derivative = activation_derivative(x, name)

        # Apply chain rule
        grad_input = grad_output * derivative

        return grad_input, None


class CustomActivation(torch.nn.Module):

    def __init__(self, name: str):
        super().__init__()

        if name not in ("relu", "sigmoid"):
            raise ValueError(
                f"Unsupported activation: {name}"
            )

        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return CustomActivationFunction.apply(
            x,
            self.name
        )