"""Read immutable CSV splits and provide image/label samples in M1."""
from pathlib import Path
import torch
from torch.utils.data import Dataset
from .config import CONFIG, CLASS_TO_LABEL
import pandas as pd
from .preprocessing import build_preprocessing
from torch.utils.data import Dataset, DataLoader
import cv2


class HorseHumanDataset(Dataset):
    """TODO: subclass torch.utils.data.Dataset in M1."""
    def __init__(self, csv_path: Path, transform=None):
            """Initialize dataset from a CSV split."""

            super().__init__()

            self.root = CONFIG.root

            # 1. Validate CSV path
            csv_path = self.root / Path(csv_path)

            if not csv_path.is_file():
                raise FileNotFoundError(
                    f"Cannot find CSV: {csv_path}"
                )

            # 2. Read metadata
            data = pd.read_csv(csv_path)

            # 3. Validate required columns
            required_columns = {
                "image_path",
                "label",
                "class_name"
            }

            missing_columns = required_columns - set(data.columns)

            if missing_columns:
                raise ValueError(
                    f"Missing required columns: {missing_columns}"
                )

            # 4. Validate dataset
            if data.empty:
                raise ValueError("Dataset is empty")

            if data[list(required_columns)].isna().any().any():
                raise ValueError("Dataset contains missing values")

            # 5. Validate binary labels
            if not data["label"].isin([0, 1]).all():
                raise ValueError("Labels must be 0 or 1")

            # Validate consistency between class and label
            expected_labels = data["class_name"].map(CLASS_TO_LABEL)

            if not expected_labels.eq(data["label"]).all():
                raise ValueError("Class names and labels do not match")

            # 6. Validate image paths
            for path in data["image_path"]:

                image_path = self.root / path

                if not image_path.is_file():
                    raise FileNotFoundError(f"Image not found: {image_path}")

            # 7. Store metadata
            self.samples = data.reset_index(drop=True)

            # 8. Set preprocessing transform
            if transform is None:
                self.transform = build_preprocessing(CONFIG.image_size)
            else:
                self.transform = transform


    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return image [3,64,64] and float binary target [1]."""
        row = self.samples.iloc[index]
        img_path = self.root/row["image_path"]
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Cannot read image: {img_path}")

        image = self.transform(image)
        label = row["label"]
        target = torch.tensor([label],dtype=torch.float32)

        return image, target


def build_loaders(config, train_transform=None, eval_transform=None) -> dict:
    """Input: config/transforms. Output: train/val/test DataLoaders.
    """
    if train_transform is None:
        train_transform = build_preprocessing(
            config.image_size
        )

    if eval_transform is None:
        eval_transform = build_preprocessing(
            config.image_size
        )

    train_dataset = HorseHumanDataset(
        csv_path=config.splits_dir / "train.csv",
        transform=train_transform
    )
    val_dataset = HorseHumanDataset(
        csv_path=config.splits_dir / "val.csv",
        transform=eval_transform
    )
    test_dataset = HorseHumanDataset(
        csv_path=config.splits_dir / "test.csv",
        transform=eval_transform
    )

    generator = torch.Generator()
    generator.manual_seed(config.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        generator=generator,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False
    )


    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }
