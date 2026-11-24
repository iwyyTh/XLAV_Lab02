"""M2 independent tests: exact pixel oracles, controlled RNG, real loader integration.

No statistical probability assertions and no lecturer demo imports. Random states
are restored after each test. All data files are read-only.
"""
import random
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

import source.augmentation as aug
from source.config import CONFIG
from source.dataset import build_loaders
from source.preprocessing import build_preprocessing


@pytest.fixture(autouse=True)
def preserve_rng():
    py_state, torch_state = random.getstate(), torch.get_rng_state()
    random.seed(42)
    yield
    random.setstate(py_state)
    torch.set_rng_state(torch_state)


@pytest.fixture(params=[3, 4], ids=['BGR','BGRA'])
def image(request):
    data = np.arange(5*7*3, dtype=np.uint8).reshape(5,7,3)
    if request.param == 4:
        data = np.concatenate((data, np.full((5,7,1),255,np.uint8)), axis=2)
    return data


def assert_tensor(value, shape=(3,64,64)):
    assert value.shape == shape
    assert value.dtype == torch.float32
    assert value.is_contiguous()
    assert torch.isfinite(value).all()
    assert 0 <= value.min() <= value.max() <= 1


class TestHorizontalFlip:
    def test_exact_coordinate_mapping_and_double_flip(self, image):
        before = image.copy()
        result = aug.horizontal_flip(image)
        assert result.shape == image.shape and result.dtype == image.dtype
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                np.testing.assert_array_equal(result[y,x], before[y,image.shape[1]-1-x])
        np.testing.assert_array_equal(aug.horizontal_flip(result), before)
        np.testing.assert_array_equal(image, before)
        assert not np.shares_memory(image,result)


class TestRandomHorizontalFlip:
    @pytest.mark.parametrize('probability,draw,flipped', [
        (0.,0.,False), (1.,0.999999,True),
        (.5,.4999,True), (.5,.5,False), (.5,.9,False),
    ])
    def test_probability_boundaries(self, image, monkeypatch, probability, draw, flipped):
        before = image.copy()
        rng = Mock(return_value=draw)
        monkeypatch.setattr(aug.random,'random',rng)
        result = aug.random_horizontal_flip(image,probability)
        expected = np.flip(before,axis=1) if flipped else before
        np.testing.assert_array_equal(result,expected)
        np.testing.assert_array_equal(image,before)

    @pytest.mark.parametrize('probability', [-.1,1.1,float('nan'),float('inf'),float('-inf')])
    def test_invalid_probability(self, probability):
        with pytest.raises(ValueError,match='Probability'):
            aug.random_horizontal_flip(np.zeros((3,3,3),np.uint8),probability)

    def test_same_python_seed_reproduces_sequence(self, image):
        random.seed(17)
        first = [aug.random_horizontal_flip(image) for _ in range(8)]
        random.seed(17)
        second = [aug.random_horizontal_flip(image) for _ in range(8)]
        for a,b in zip(first,second):
            np.testing.assert_array_equal(a,b)


class TestRotation:
    def test_zero_identity(self, image):
        before = image.copy()
        actual = aug.rotate_image(image,0.)
        np.testing.assert_array_equal(actual,before)
        np.testing.assert_array_equal(image,before)
        assert actual.dtype == image.dtype and actual.shape == image.shape

    @pytest.mark.parametrize('angle,k', [(90,1),(-90,-1),(180,2)])
    @pytest.mark.parametrize('channels', [3,4])
    def test_square_quarter_turn_exact_pixels(self, channels, angle, k):
        image = np.arange(5*5*channels,dtype=np.uint8).reshape(5,5,channels)
        before = image.copy()
        # np.rot90 is independent of OpenCV's affine transform implementation.
        expected = np.rot90(before,k,axes=(0,1))
        np.testing.assert_array_equal(aug.rotate_image(image,angle),expected)
        np.testing.assert_array_equal(image,before)

    def test_non_square_90_reflect101_border_exact_pixels(self):
        image = np.arange(3*5*3,dtype=np.uint8).reshape(3,5,3)
        expected = np.empty_like(image)
        # Center=(2,1). Inverse +90 mapping: source=(3-y,x-1).
        # REFLECT_101 maps source row -1->1 and 3->1, without edge repetition.
        reflected_rows = [1,0,1,2,1]
        for y in range(3):
            for x in range(5):
                expected[y,x] = image[reflected_rows[x],3-y]
        result = aug.rotate_image(image,90.)
        np.testing.assert_array_equal(result,expected)
        assert result.shape == (3,5,3) and result.dtype == np.uint8

    @pytest.mark.parametrize('angle', [-37.,-15.,15.,37.])
    def test_signed_angles_preserve_shape_dtype_alpha_and_input(self, image, angle):
        before = image.copy()
        result = aug.rotate_image(image,angle)
        assert result.shape == image.shape and result.dtype == image.dtype
        np.testing.assert_array_equal(image,before)
        if image.shape[2] == 4:
            assert np.all(result[:,:,3] == 255)

    def test_border_does_not_create_black_or_transparent_corners(self):
        image = np.full((9,13,4),(25,80,170,255),np.uint8)
        result = aug.rotate_image(image,31.)
        # Reflecting a constant image must preserve every pixel including corners.
        np.testing.assert_array_equal(result,image)

    def test_rotation_geometry_and_interpolation_arguments(self, monkeypatch):
        real_warp = cv2.warpAffine
        spy = Mock(wraps=real_warp)
        monkeypatch.setattr(aug.cv2,'warpAffine',spy)
        aug.rotate_image(np.zeros((3,5,3),np.uint8),90.)
        args,kwargs = spy.call_args
        np.testing.assert_allclose(args[1],[[0,1,1],[-1,0,3]],atol=1e-12)
        assert args[2] == (5,3)
        assert kwargs['flags'] == cv2.INTER_LINEAR
        assert kwargs['borderMode'] == cv2.BORDER_REFLECT_101

    @pytest.mark.parametrize('angle', [float('nan'),float('inf'),float('-inf')])
    def test_nonfinite_angle_rejected(self, angle):
        with pytest.raises(ValueError,match='finite'):
            aug.rotate_image(np.zeros((3,5,3),np.uint8),angle)


class TestRandomRotation:
    def test_zero_max_angle_is_identity(self, image):
        before = image.copy()
        np.testing.assert_array_equal(aug.random_rotation(image,0),before)
        np.testing.assert_array_equal(image,before)

    @pytest.mark.parametrize('sampled', [-15.,0.,15.])
    def test_uniform_bounds_and_forwarded_angle(self, monkeypatch,sampled):
        image = np.zeros((3,5,3),np.uint8)
        sample = Mock(return_value=sampled)
        rotate = Mock(return_value=image.copy())
        monkeypatch.setattr(aug.random,'uniform',sample)
        monkeypatch.setattr(aug,'rotate_image',rotate)
        aug.random_rotation(image,15.)
        sample.assert_called_once_with(-15.,15.)
        assert rotate.call_args.args[0] is image
        assert rotate.call_args.args[1] == sampled

    def test_fixed_seed_samples_are_within_requested_interval(self, monkeypatch):
        image = np.zeros((3,5,3),np.uint8)
        rotate = Mock(return_value=image)
        monkeypatch.setattr(aug,'rotate_image',rotate)
        random.seed(123)
        for _ in range(20):
            aug.random_rotation(image,7.)
        angles = [call.args[1] for call in rotate.call_args_list]
        assert len(angles) == 20
        assert all(-7 <= angle <= 7 for angle in angles)

    @pytest.mark.parametrize('maximum', [-1.,float('nan'),float('inf'),float('-inf')])
    def test_invalid_max_angle(self, maximum):
        with pytest.raises(ValueError,match='finite and non-negative'):
            aug.random_rotation(np.zeros((3,5,3),np.uint8),maximum)

    def test_same_python_seed_reproduces_rotation(self, image):
        before = image.copy()
        random.seed(23)
        first = [aug.random_rotation(image) for _ in range(4)]
        random.seed(23)
        second = [aug.random_rotation(image) for _ in range(4)]
        for a,b in zip(first,second):
            np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(image,before)


class TestPipeline:
    def test_tensor_contract_input_preserved_and_seed_reproducible(self,image):
        before = image.copy()
        transform = aug.build_augmentation(CONFIG)
        random.seed(42)
        first = transform(image)
        random.seed(42)
        second = transform(image)
        assert_tensor(first)
        assert torch.equal(first,second)
        np.testing.assert_array_equal(image,before)

    def test_flip_rotation_preprocessing_order_and_parameters(self,monkeypatch):
        original = np.zeros((3,5,3),np.uint8)
        flipped,rotated = original+1,original+2
        output = torch.ones((3,64,64))
        events = []
        def flip(image,probability):
            assert image is original and probability == .5
            events.append('flip')
            return flipped
        def rotate(image,max_angle):
            assert image is flipped and max_angle == 15.
            events.append('rotation')
            return rotated
        def preprocess(image):
            assert image is rotated
            events.append('preprocessing')
            return output
        factory = Mock(return_value=preprocess)
        monkeypatch.setattr(aug,'random_horizontal_flip',flip)
        monkeypatch.setattr(aug,'random_rotation',rotate)
        monkeypatch.setattr(aug,'build_preprocessing',factory)
        transform = aug.build_augmentation(CONFIG)
        factory.assert_called_once_with(CONFIG.image_size)
        assert transform(original) is output
        assert events == ['flip','rotation','preprocessing']

    def test_controlled_flip_and_zero_rotation_produce_correct_rgb(self,monkeypatch):
        from dataclasses import replace
        config = replace(CONFIG,image_size=(1,2))
        image = np.array([[[0,0,255,255],[255,0,0,255]]],np.uint8)
        monkeypatch.setattr(aug.random,'random',lambda:0.)
        monkeypatch.setattr(aug.random,'uniform',lambda low,high:0.)
        result = aug.build_augmentation(config)(image)
        expected = torch.tensor([[[0.,1.]],[[0.,0.]],[[1.,0.]]])
        assert torch.equal(result,expected)


class TestDataLoaderIntegration:
    def test_real_batches_train_only_eval_deterministic_and_labels(self,monkeypatch):
        """Real first batches: routing, exact labels, deterministic eval, no split writes."""
        import source.dataset as dataset_module
        paths = []
        reader = cv2.imread
        def record(path,flags):
            paths.append(Path(path).resolve())
            return reader(path,flags)
        monkeypatch.setattr(dataset_module.cv2,'imread',record)
        train = Mock(wraps=aug.build_augmentation(CONFIG))
        evaluate = Mock(wraps=build_preprocessing(CONFIG.image_size))
        loaders = build_loaders(CONFIG,train_transform=train,eval_transform=evaluate)
        assert train.call_count == evaluate.call_count == 0
        train_images,train_targets = next(iter(loaders['train']))
        assert_tensor(train_images,(32,3,64,64))
        assert train_targets.shape == (32,1) and train_targets.dtype == torch.float32
        rows = pd.read_csv(CONFIG.splits_dir/'train.csv')
        labels = {(CONFIG.root/r.image_path).resolve():int(r.label) for r in rows.itertuples()}
        assert train_targets.flatten().tolist() == [labels[p] for p in paths]
        assert train.call_count == 32 and evaluate.call_count == 0
        py_state = random.getstate()
        for split in ('val','test'):
            x,y = next(iter(loaders[split]))
            z,t = next(iter(loaders[split]))
            assert_tensor(x,(32,3,64,64))
            assert y.shape == (32,1) and y.dtype == torch.float32
            assert torch.equal(x,z) and torch.equal(y,t)
            expected_labels = pd.read_csv(CONFIG.splits_dir/f'{split}.csv').label.iloc[:32].tolist()
            assert y.flatten().tolist() == expected_labels
            assert train.call_count == 32
            assert random.getstate() == py_state
        assert evaluate.call_count == 128

    def test_real_train_first_batch_reproducible_when_both_rngs_seeded(self):
        # Loader seeds torch shuffle; augmentation separately uses Python random.
        first = build_loaders(CONFIG,train_transform=aug.build_augmentation(CONFIG))['train']
        second = build_loaders(CONFIG,train_transform=aug.build_augmentation(CONFIG))['train']
        random.seed(CONFIG.seed)
        a,y = next(iter(first))
        random.seed(CONFIG.seed)
        b,z = next(iter(second))
        assert torch.equal(a,b) and torch.equal(y,z)
