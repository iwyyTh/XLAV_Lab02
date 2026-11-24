"""Implement optimization and validation loops in M6."""
from pathlib import Path
import torch
import json
from .metrics import binary_metrics


def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device: str
) -> dict:
    """Train the model for one epoch."""

    # Set model to training mode

    model = model.train()

    total_loss = 0.0
    total_samples = 0

    all_probabilities = []
    all_targets = []

    for images, targets in loader:

        # Move images and targets to device
        images = images.to(device)
        targets = targets.to(device)

        # Reset gradients
        optimizer.zero_grad()


        # Forward propagation
        logits = model(images)


        # Calculate loss
        loss = criterion(logits,targets)


        # Check that loss is finite
        if not torch.isfinite(loss):
            raise FloatingPointError("Loss is not finite")


        # Backward propagation
        loss.backward()

        # Update model parameters
        optimizer.step()


        # Accumulate sample-weighted loss
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size

        total_samples += batch_size

        # Convert detached logits to probabilities
        probabilities = torch.sigmoid(
            logits.detach()
        )


        # Collect probabilities and targets on CPU
        all_probabilities.append(
            probabilities.reshape(-1).cpu()
        )
        all_targets.append(
            targets.detach().reshape(-1).cpu()
        )
    # Reject empty DataLoader
    if total_samples == 0:
        raise ValueError("empty DataLoader")

    # Calculate average epoch loss
    epoch_loss = total_loss / total_samples

    # Concatenate probabilities and targets
    probabilities = torch.cat(all_probabilities)

    targets = torch.cat(all_targets)

    # Calculate binary metrics
    metrics = binary_metrics(
        targets,
        probabilities
    )

    # Return epoch statistics

    return {
        "loss": epoch_loss,
        **metrics
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device: str
) -> dict:
    """Evaluate the model without updating parameters."""

    # Set model to evaluation mode
    model = model.eval()

    total_loss = 0.0
    total_samples = 0

    all_probabilities = []
    all_targets = []

    for images, targets in loader:

        # Move images and targets to device
        images = images.to(device)
        targets = targets.to(device)

        # Forward propagation
        logits = model(images)
        # Calculate loss
        loss = criterion(logits,targets)

        # Validate finite loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Loss is not finite")

        # Accumulate sample-weighted loss
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # Convert logits to probabilities
        probabilities = torch.sigmoid(
            logits.detach()
        )

        # Collect probabilities and targets on CPU
        all_probabilities.append(
            probabilities.reshape(-1).cpu()
        )
        all_targets.append(
            targets.detach().reshape(-1).cpu()
        )

    # Reject empty DataLoader
    if total_samples == 0:
        raise ValueError("empty DataLoader")

    # Calculate average epoch loss
    epoch_loss = total_loss / total_samples

    # Concatenate probabilities and targets
    probabilities = torch.cat(all_probabilities)

    targets = torch.cat(all_targets)

    # Calculate binary metrics
    metrics = binary_metrics(
        targets,
        probabilities
    )

    return {
        "loss": epoch_loss,
        **metrics
    }

def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device: str,
    epochs: int,
    checkpoint_path: Path,
    history_path: Path
) -> dict:
    """
    Train the model for multiple epochs.

    Save the best checkpoint according to
    validation loss and record training history.
    """

    # Validate epochs
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    # Create output directories
    checkpoint_path = Path(checkpoint_path)
    history_path = Path(history_path)

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    history_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    best_val_loss = float("inf")
    best_epoch = 0

    history = {
        "train": [],
        "val": []
    }

    for epoch in range(1, epochs + 1):

        train_stats = train_epoch(model,train_loader,optimizer,criterion,device)

        # Call evaluate()

        val_stats = evaluate(model,val_loader,criterion,device)

        # Append epoch statistics to history
        history["train"].append({
            "epoch": epoch,
            **train_stats
        })

        history["val"].append({
            "epoch": epoch,
            **val_stats
        })
        # Compare current validation loss
        # with best validation loss

        if val_stats["loss"] < best_val_loss:

            # Update best_val_loss
            best_val_loss = val_stats["loss"]
            # Update best_epoch
            best_epoch = epoch
            # Save checkpoint
            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                checkpoint_path,
            )
           

        # Print training progress
        print(
            f"Epoch {epoch}/{epochs} | "
            f"train loss: {train_stats['loss']:.4f} | "
            f"val loss: {val_stats['loss']:.4f} | "
            f"val F1: {val_stats['f1']:.4f}"
        )

    # Save best epoch information to history
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val_loss

    # Save history to JSON
    with history_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2, ensure_ascii=False)

    return history
