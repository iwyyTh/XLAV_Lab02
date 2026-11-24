"""Build the binary MLP in M5."""
from pathlib import Path
import torch


class MLP(torch.nn.Module):
    def __init__(self, input_dim: int = 12288, hidden_dims=(128, 64), activation: str = "relu"):
        """Input: layer sizes/activation. Output: MLP with a single logit per image.

        TODO: flatten -> hidden layers -> linear output; no sigmoid before BCE loss.
        """
        super().__init__()
        raise NotImplementedError("M5")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input: [B,3,64,64]. Output: logits [B,1]. TODO: implement layers."""
        raise NotImplementedError("M5")
