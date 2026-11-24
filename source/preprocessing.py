"""Define deterministic RGB resizing and tensor conversion in M1."""
import torch
import cv2
import numpy as np


def build_preprocessing(image_size: tuple[int, int] = (64, 64)):
    """Return deterministic uint8 OpenCV BGR/BGRA -> RGB float32 [3,H,W].

    image_size is (height, width); output is contiguous and scaled to [0,1].
    BGRA transparency is composited onto white before resize, preserving the
    previous Pillow policy. Opaque alpha leaves RGB values unchanged.
    Reject empty, grayscale, non-uint8 or non-3/4-channel arrays. No flatten
    or augmentation is performed; the input array is never modified.
    """
    if (not isinstance(image_size, (tuple, list)) or len(image_size) != 2
            or any(isinstance(n, bool) or not isinstance(n, (int, np.integer))
                   or n <= 0 for n in image_size)):
        raise ValueError("image_size must contain positive integers (height, width)")
    height, width = image_size

    def transform(image: np.ndarray) -> torch.Tensor:
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a NumPy uint8 BGR/BGRA array, not None or PIL")
        if image.ndim != 3 or image.shape[2] not in (3, 4) or image.size == 0:
            raise ValueError("image must be non-empty HWC with 3 BGR or 4 BGRA channels")
        if image.dtype != np.uint8:
            raise TypeError("image must have dtype uint8 with pixels in [0,255]")
        if image.shape[2] == 4:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            if np.any(image[:, :, 3] != 255):
                alpha = image[:, :, 3:4].astype(np.float32) / 255.0
                # Composite before resize, rounding to the original 8-bit scale.
                rgb = np.rint(rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # OpenCV dsize reverses the project's (height, width) convention.
        image = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)

        image = image.astype(np.float32)
        image = image / 255.0

        img_tensor = torch.from_numpy(image)
        img_tensor = img_tensor.permute(2, 0, 1).contiguous()

        return img_tensor

    return transform
