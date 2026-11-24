import json
import cv2
import torch

from pathlib import Path
from dataclasses import replace, fields

from .config import CONFIG
from .model import MLP
from .preprocessing import build_preprocessing
from .dataset import build_loaders
from .trainer import evaluate

def load_run_config(run_name: str, device=None):

    config_path = (
        CONFIG.results_dir
        / "configs"
        / f"{run_name}.json"
    )

    with config_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    valid_fields = {
        field.name for field in fields(CONFIG)
    }

    config_values = {
        key: value
        for key, value in data.items()
        if key in valid_fields
    }

    # Use the current project location
    config_values["root"] = CONFIG.root

    config_values["image_size"] = tuple(
        config_values["image_size"]
    )

    config_values["device"] = (
        device if device is not None
        else CONFIG.device
    )

    return replace(CONFIG, **config_values)

def load_model(checkpoint_path: Path, config) -> MLP:

    model = MLP(
        input_dim=config.input_dim,
        hidden_dims=config.hidden_dims,
        activation=config.activation
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(config.device)

    model.eval()

    return model

@torch.no_grad()
def predict(
    image_path: Path,
    checkpoint_path: Path,
    config
) -> dict:

    # Load trained model

    model = load_model(checkpoint_path,config)

    # Read image using OpenCV IMREAD_UNCHANGED

    image = cv2.imread(str(image_path),cv2.IMREAD_UNCHANGED)

    # Reject unreadable image
    if image is None:
        raise ValueError("unreadable image")

    # Build preprocessing

    preprocessing = build_preprocessing(config.image_size)

    # Apply preprocessing

    tensor = preprocessing(image)

    # Add batch dimension and move to device
    tensor = tensor.unsqueeze(0).to(config.device)

    # Forward propagation

    logits = model(tensor)

    # Convert logit to human probability

    probability = torch.sigmoid(logits).item()

    # Convert probability to predicted label
    label = 1 if probability >= 0.5 else 0

    return {
        "label": label,
        "class_name": "human" if label == 1 else "horse",
        "human_probability": probability
    }

def evaluate_checkpoint(
    checkpoint_path: Path,
    config,
    split: str = "val"
) -> dict:

    if split not in ("val", "test"):
        raise ValueError(
            "Split must be val or test"
        )

    model = load_model(
        checkpoint_path,
        config
    )

    preprocessing = build_preprocessing(
        config.image_size
    )

    loaders = build_loaders(
        config,
        train_transform=preprocessing,
        eval_transform=preprocessing
    )

    criterion = torch.nn.BCEWithLogitsLoss()

    return evaluate(
        model,
        loaders[split],
        criterion,
        config.device
    )