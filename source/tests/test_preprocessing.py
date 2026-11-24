"""OpenCV preprocessing contract; no Dataset/DataLoader implementation assumed."""
import cv2
import numpy as np
import pytest
import torch

from source.config import CONFIG
from source.preprocessing import build_preprocessing


@pytest.mark.parametrize('channels', [3, 4])
def test_channel_order(channels):
    # BGR primaries -> expected RGB primaries; no resizing at this resolution.
    image = np.array([[[0, 0, 255], [0, 255, 0], [255, 0, 0]]], dtype=np.uint8)
    if channels == 4:
        image = np.concatenate([image, np.full((1, 3, 1), 255, dtype=np.uint8)], axis=2)
    output = build_preprocessing((1, 3))(image)
    expected = torch.tensor([[[1., 0., 0.]], [[0., 1., 0.]], [[0., 0., 1.]]])
    assert torch.equal(output, expected)


@pytest.mark.parametrize('size', [(64, 64), (32, 64)])
def test_tensor_contract_and_determinism(size):
    image = np.arange(7 * 11 * 3, dtype=np.uint8).reshape(7, 11, 3)
    original = image.copy()
    transform = build_preprocessing(size)
    output = transform(image)
    assert output.shape == (3, *size)
    assert output.dtype == torch.float32
    assert output.is_contiguous()
    assert torch.isfinite(output).all()
    assert 0 <= output.min() <= output.max() <= 1
    assert torch.equal(output, transform(image))
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize('value', [0, 255])
@pytest.mark.parametrize('channels', [3, 4])
def test_black_white(value, channels):
    image = np.full((3, 5, channels), value, dtype=np.uint8)
    if channels == 4:
        image[:, :, 3] = 255
    output = build_preprocessing()(image)
    assert torch.all(output == value / 255)


def test_transparency_composited_on_white():
    image = np.array([[[0, 0, 255, 0], [0, 0, 255, 128], [0, 0, 255, 255]]], dtype=np.uint8)
    original = image.copy()
    result = build_preprocessing((1, 3))(image)
    expected = torch.tensor([[[255, 255, 255]], [[255, 127, 0]], [[255, 127, 0]]], dtype=torch.float32) / 255
    assert torch.equal(result, expected)
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize('image,error', [
    (None, TypeError), ('invalid', TypeError),
    (np.empty((0, 3, 3), dtype=np.uint8), ValueError),
    (np.zeros((2, 2), dtype=np.uint8), ValueError),
    (np.zeros((2, 2, 1), dtype=np.uint8), ValueError),
    (np.zeros((2, 2, 5), dtype=np.uint8), ValueError),
    (np.zeros((2, 2, 3), dtype=np.uint16), TypeError),
    (np.full((2, 2, 3), np.nan), TypeError),
    (np.full((2, 2, 3), np.inf), TypeError),
])
def test_invalid_image(image, error):
    with pytest.raises(error, match='image'):
        build_preprocessing()(image)


@pytest.mark.parametrize('size', [(0, 64), (32, -1), (32.5, 64), (True, 64), (64,), None])
def test_invalid_size(size):
    with pytest.raises(ValueError, match='image_size'):
        build_preprocessing(size)


def test_noncontiguous_input():
    image = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)[:, ::-1]
    transform = build_preprocessing((3, 5))
    assert torch.equal(transform(image), transform(image.copy()))
    assert transform(image).is_contiguous()


def test_real_png_reading():
    path = next(CONFIG.raw_dir.rglob('*.png'), None)
    if path is None:
        pytest.skip('Real dataset is not available in this checkout')
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert image is not None, f'OpenCV could not read {path}'
    assert image.shape == (300, 300, 4)
    assert image.dtype == np.uint8
    assert np.all(image[:, :, 3] == 255)
    assert build_preprocessing()(image).shape == (3, 64, 64)
