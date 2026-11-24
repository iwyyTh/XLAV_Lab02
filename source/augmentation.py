"""Implement train-only image transformations in M2."""
from pathlib import Path
import torch
import cv2
import numpy as np
import random
import math
from .preprocessing import build_preprocessing


def horizontal_flip(image: np.ndarray) -> np.ndarray:
    return image[:, ::-1, :].copy()

def random_horizontal_flip(
    image: np.ndarray,
    probability: float = 0.5
) -> np.ndarray:
    """Randomly flip an image horizontally."""

    # Validate probability
    if not 0 <= probability <= 1:
        raise ValueError("Probability must be in [0, 1]")

    # Generate a random number
    random_num = random.random()

    # Apply horizontal flip with the given probability
    if random_num < probability:
        image = horizontal_flip(image)

    return image

def rotate_image(
    image: np.ndarray,
    angle: float
) -> np.ndarray:
    """Rotate an image around its center."""

    if not math.isfinite(angle):
        raise ValueError("Angle must be finite")

    # Step 1: Get image dimensions
    height, width = image.shape[:2]

    # Step 2: Calculate rotation center
    center = (
        (width - 1) / 2,
        (height - 1) / 2
    )

    # Step 3: Construct rotation matrix
    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    # Step 4: Apply affine transformation
    result = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    return result

def random_rotation(
    image: np.ndarray,
    max_angle: float = 15.0
) -> np.ndarray:
    """Rotate an image by a randomly sampled angle."""

    # Step 1: Validate maximum angle
    if not math.isfinite(max_angle) or max_angle < 0:
        raise ValueError(
            "max_angle must be finite and non-negative"
        )

    # Step 2: Generate a random angle
    angle = random.uniform(-max_angle, max_angle)

    # Step 3: Rotate image
    return rotate_image(image, angle)

def build_augmentation(config):
    """Build train-only augmentation and preprocessing."""

    preprocessing = build_preprocessing(
        config.image_size
    )

    def transform(image: np.ndarray) -> torch.Tensor:

        # Step 1: Random horizontal flip
        image = random_horizontal_flip(
            image,
            probability=0.5
        )

        # Step 2: Random rotation
        image = random_rotation(
            image,
            max_angle=15.0
        )

        # Step 3: Apply deterministic preprocessing
        tensor = preprocessing(image)

        return tensor

    return transform