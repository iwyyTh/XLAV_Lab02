"""Central configuration; paths are relative to this project, never to the host."""
from dataclasses import dataclass, field
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
CLASS_TO_LABEL = {"horses": 0, "humans": 1}

@dataclass(frozen=True)
class Config:
    """Defaults for later experiments; binary logits use one output unit."""
    root: Path = ROOT
    seed: int = 42
    image_size: tuple[int, int] = (64, 64)
    image_channels: int = 3
    num_classes: int = 2
    hidden_dims: list[int] = field(default_factory=lambda: [128, 64])
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 30
    activation: str = "relu"
    initialization: str = "he_normal"
    optimizer: str = "Adam"
    loss: str = "BCEWithLogitsLoss"
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    num_workers: int = 0

    @property
    def input_dim(self) -> int:
        return self.image_size[0] * self.image_size[1] * self.image_channels

    @property
    def raw_dir(self) -> Path:
        return self.root / "data/raw"

    @property
    def splits_dir(self) -> Path:
        return self.root / "data/splits"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

CONFIG = Config()
