import argparse
import json
import random

from dataclasses import replace, asdict

import numpy as np
import torch

from .config import CONFIG
from .dataset import build_loaders
from .preprocessing import build_preprocessing
from .augmentation import build_augmentation
from .model import MLP, initialize_model
from .trainer import fit

def parse_args():

    parser = argparse.ArgumentParser(
        description="Horse or Human MLP Training"
    )

    parser.add_argument(
        "--activation",
        choices=["relu", "sigmoid"],
        default=CONFIG.activation
    )

    parser.add_argument(
        "--initialization",
        choices=[
            "normal",
            "fan_in_normal",
            "xavier_normal",
            "he_normal"
        ],
        default=CONFIG.initialization
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=CONFIG.epochs
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=CONFIG.learning_rate
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=CONFIG.batch_size
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=CONFIG.seed
    )

    parser.add_argument(
        "--no-augmentation",
        action="store_true"
    )

    parser.add_argument(
        "--run-name",
        default="baseline"
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=CONFIG.device
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:

    # Parse command line arguments
    args = parse_args()

    # Create a new config using dataclasses.replace()
    config = replace(
        CONFIG,
        activation=args.activation,
        initialization=args.initialization,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device
    )

    if config.epochs <= 0 or config.batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if not np.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")

    # Set random seeds
    set_seed(config.seed)
    # Build evaluation transform
    eval_transform = build_preprocessing(config.image_size)

    # Build train transform
    # If --no-augmentation is enabled,
    # use preprocessing only
    if args.no_augmentation:
        train_transform = eval_transform
    else:
        train_transform = build_augmentation(config)

    # Build DataLoaders
    loaders = build_loaders(config,train_transform,eval_transform)

    # Create MLP
    model = MLP(config.input_dim,config.hidden_dims,config.activation)

    # Initialize model weights
    initialize_model(model,config.initialization)
    # Move model to device
    model.to(config.device)
    # Create BCEWithLogitsLoss
    criterion = torch.nn.BCEWithLogitsLoss()

    # Create Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Define output paths

    checkpoint_path = (
        config.results_dir
        / "checkpoints"
        / f"{args.run_name}.pt"
    )

    history_path = (
        config.results_dir
        / "histories"
        / f"{args.run_name}.json"
    )

    # Call fit()

    history = fit(
        model,
        loaders["train"],
        loaders["val"],
        optimizer,
        criterion,
        config.device,
        config.epochs,
        checkpoint_path,
        history_path
    )

    # Save experiment configuration
    config_path = config.results_dir / "configs" / f"{args.run_name}.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    experiment_config = asdict(config)
    experiment_config["root"] = str(config.root)
    experiment_config["run_name"] = args.run_name
    experiment_config["augmentation_enabled"] = not args.no_augmentation
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(experiment_config, file, indent=2, ensure_ascii=False)

    # Print experiment summary
    print(f"Run: {args.run_name}")
    print(f"Best epoch: {history['best_epoch']}")
    print(f"Best validation loss: {history['best_val_loss']:.4f}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"History: {history_path}")
    print(f"Configuration: {config_path}")


if __name__ == "__main__":
    main()
