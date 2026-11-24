"""Read immutable CSV splits and provide image/label samples in M1."""
from pathlib import Path
import torch


class HorseHumanDataset:
    """TODO: subclass torch.utils.data.Dataset in M1."""
    def __init__(self, csv_path: Path, transform=None):
        """Input: split CSV and optional transform. Output: dataset instance."""
        # TODO: validate CSV, resolve paths from project root, retain labels.
        raise NotImplementedError("M1")

    def __len__(self) -> int:
        """Return the number of samples."""
        raise NotImplementedError("M1")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return image [3,64,64] and float binary target [1]."""
        # TODO: open safely, convert RGB, apply transform, return target.
        raise NotImplementedError("M1")


def build_loaders(config, train_transform=None, eval_transform=None) -> dict:
    """Input: config/transforms. Output: train/val/test DataLoaders.

    TODO: seeded train shuffling only; workers=0; never augment val/test.
    """
    raise NotImplementedError("M1")
