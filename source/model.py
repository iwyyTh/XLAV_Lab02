"""Build the binary MLP in M5."""
from pathlib import Path
import torch
from .activations import CustomActivation
from .initialization import initialize_weights

class MLP(torch.nn.Module):
    def __init__(
        self,
        input_dim: int = 12288,
        hidden_dims=(128, 64),
        activation: str = "relu"
    ):
        super().__init__()
        self.input_dim = input_dim
        layers = []

        previous_dim = input_dim

        for hidden_dim in hidden_dims:

            # Create nn.Linear
            layer = torch.nn.Linear(previous_dim,hidden_dim)
            # Append Linear layer
            layers.append(layer)
            # Append CustomActivation
            layers.append(CustomActivation(activation))
            # Update previous_dim
            previous_dim = hidden_dim


        layers.append(torch.nn.Linear(previous_dim,1))

        self.layers = torch.nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        x = x.flatten(start_dim=1)

        logits = self.layers(x)
      
        return logits

def initialize_model(
    model: MLP,
    method: str = "he_normal"
) -> None:
    linear_layers = [
        layer
        for layer in model.modules()
        if isinstance(layer, torch.nn.Linear)
    ]
    for layer in linear_layers[:-1]:
        initialize_weights(layer,method)

    initialize_weights(linear_layers[-1],"xavier_normal")