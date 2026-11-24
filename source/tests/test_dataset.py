"""Independent M1 contracts. Synthetic unit tests + one full real-data traversal.

P/D/L IDs refer to the requested checklist. No reference demos are imported.
Temporary fixtures patch only the module CONFIG binding, never files/config on disk.
"""
from collections import Counter
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import RandomSampler, SequentialSampler

import source.dataset as dataset_module
from source.config import CONFIG, CLASS_TO_LABEL
from source.dataset import HorseHumanDataset, build_loaders
from source.preprocessing import build_preprocessing


@pytest.fixture
def synthetic_project(tmp_path, monkeypatch):
    """Six distinct tiny images, all paths relative to an isolated project root."""
    config = replace(CONFIG, root=tmp_path, batch_size=2)
    monkeypatch.setattr(dataset_module, 'CONFIG', config)
    config.splits_dir.mkdir(parents=True)
    rows = []
    for i in range(6):
        label = i % 2
        cls = 'horses' if label == 0 else 'humans'
        p = tmp_path / 'data/raw' / cls / f'{i}.png'
        p.parent.mkdir(parents=True, exist_ok=True)
        image = np.full((5, 7, 4), (i * 30, 10, 200, 255), np.uint8)
        assert cv2.imwrite(str(p), image)
        rows.append(dict(image_path=p.relative_to(tmp_path).as_posix(), label=label, class_name=cls))
    for split in ('train', 'val', 'test'):
        pd.DataFrame(rows).to_csv(config.splits_dir / f'{split}.csv', index=False)
    return config, rows


@pytest.fixture
def metadata_csv(synthetic_project):
    config, rows = synthetic_project
    def write(rows_override=None, columns=None):
        data = pd.DataFrame(rows if rows_override is None else rows_override)
        if columns is not None:
            data = data.reindex(columns=columns)
        p = config.splits_dir / 'edge.csv'
        data.to_csv(p, index=False)
        return p
    return write


def assert_image(tensor, shape):
    assert tensor.shape == shape
    assert tensor.dtype == torch.float32
    assert tensor.is_contiguous()
    assert torch.isfinite(tensor).all()
    assert 0 <= tensor.min() <= tensor.max() <= 1


class TestPreprocessing:
    @pytest.mark.parametrize('size', [(64, 64), (32, 64)])
    def test_p01_p05_p10_p12_tensor_contract(self, size):
        image = np.arange(7 * 11 * 3, dtype=np.uint8).reshape(7, 11, 3)
        before = image.copy()
        transform = build_preprocessing(size)
        actual = transform(image)
        assert_image(actual, (3, *size))
        assert torch.equal(actual, transform(image))
        np.testing.assert_array_equal(image, before)

    @pytest.mark.parametrize('channels', [3, 4])
    @pytest.mark.parametrize('value', [0, 255])
    def test_p06_p07_black_white(self, channels, value):
        image = np.full((4, 7, channels), value, np.uint8)
        if channels == 4:
            image[:, :, 3] = 255
        assert torch.all(build_preprocessing()(image) == value / 255)

    @pytest.mark.parametrize('channels', [3, 4])
    def test_p08_p09_primary_colors(self, channels):
        image = np.array([[[0, 0, 255], [0, 255, 0], [255, 0, 0]]], np.uint8)
        if channels == 4:
            image = np.concatenate((image, np.full((1, 3, 1), 255, np.uint8)), axis=2)
        expected = torch.eye(3).reshape(3, 1, 3)
        assert torch.equal(build_preprocessing((1, 3))(image), expected)

    def test_p09_p12_white_alpha_composite_without_mutation(self):
        image = np.array([[[0, 0, 255, 0], [0, 0, 255, 128], [0, 0, 255, 255]]], np.uint8)
        before = image.copy()
        expected = torch.tensor([[[255,255,255]], [[255,127,0]], [[255,127,0]]], dtype=torch.float32) / 255
        assert torch.equal(build_preprocessing((1, 3))(image), expected)
        np.testing.assert_array_equal(image, before)

    @pytest.mark.parametrize('image,error', [
        (None, TypeError), ('not an image', TypeError),
        (np.empty((0, 2, 3), np.uint8), ValueError),
        (np.zeros((2, 2), np.uint8), ValueError),
        (np.zeros((2, 2, 1), np.uint8), ValueError),
        (np.zeros((2, 2, 5), np.uint8), ValueError),
        (np.zeros((2, 2, 3), np.uint16), TypeError),
        (np.full((2, 2, 3), np.nan), TypeError),
        (np.full((2, 2, 3), np.inf), TypeError),
    ])
    def test_p11_invalid_image(self, image, error):
        with pytest.raises(error, match='image'):
            build_preprocessing()(image)

    @pytest.mark.parametrize('size', [(0,64), (32,-1), (1.5,64), (True,64), (64,), None])
    def test_p11_invalid_size(self, size):
        with pytest.raises(ValueError, match='image_size'):
            build_preprocessing(size)


class TestDataset:
    @pytest.mark.parametrize('split,count', [('train',845), ('val',182), ('test',256)])
    def test_d01_d08_d17_d21_real_metadata_and_sample(self, split, count):
        path = CONFIG.splits_dir / f'{split}.csv'
        rows = pd.read_csv(path)
        ds = HorseHumanDataset(path)
        assert len(ds) == len(rows) == count
        for cls, label in zip(rows.class_name, rows.label):
            assert CLASS_TO_LABEL[cls] == label
        # Verify both labels on real images; exhaustive access happens in L11.
        for label in (0, 1):
            index = rows.index[rows.label == label][0]
            image, target = ds[int(index)]
            assert_image(image, (3,64,64))
            assert target.shape == (1,) and target.dtype == torch.float32
            assert target.item() == label

    def test_d09_missing_csv(self, tmp_path):
        with pytest.raises(FileNotFoundError, match='Cannot find CSV'):
            HorseHumanDataset(tmp_path / 'missing.csv')

    @pytest.mark.parametrize('column', ['image_path','label','class_name'])
    def test_d10_missing_column(self, metadata_csv, column):
        columns = [c for c in ['image_path','label','class_name'] if c != column]
        with pytest.raises(ValueError, match='Missing required columns'):
            HorseHumanDataset(metadata_csv(columns=columns))

    @pytest.mark.parametrize('header', [True, False])
    def test_d11_empty_csv(self, synthetic_project, header):
        config, _ = synthetic_project
        p = config.splits_dir / 'empty.csv'
        p.write_text('image_path,label,class_name\n' if header else '')
        # pandas EmptyDataError is a ValueError; both empty forms must reject.
        with pytest.raises(ValueError):
            HorseHumanDataset(p)

    @pytest.mark.parametrize('column', ['image_path','label','class_name'])
    def test_d12_missing_value(self, synthetic_project, metadata_csv, column):
        _, rows = synthetic_project
        rows[0][column] = None
        with pytest.raises(ValueError, match='missing values'):
            HorseHumanDataset(metadata_csv(rows))

    @pytest.mark.parametrize('label', [-1, 2, 0.5, 'invalid'])
    def test_d13_invalid_label(self, synthetic_project, metadata_csv, label):
        _, rows = synthetic_project
        rows[0]['label'] = label
        with pytest.raises(ValueError, match='Labels must be 0 or 1'):
            HorseHumanDataset(metadata_csv(rows))

    @pytest.mark.parametrize('class_name', ['humans','unknown'])
    def test_d14_class_label_mismatch(self, synthetic_project, metadata_csv, class_name):
        _, rows = synthetic_project
        rows[0]['class_name'] = class_name  # label remains 0
        with pytest.raises(ValueError, match='Class names and labels do not match'):
            HorseHumanDataset(metadata_csv(rows))

    def test_d15_missing_image(self, synthetic_project, metadata_csv):
        _, rows = synthetic_project
        rows[0]['image_path'] = 'data/raw/missing.png'
        with pytest.raises(FileNotFoundError, match='missing.png'):
            HorseHumanDataset(metadata_csv(rows))

    def test_d16_unreadable_image(self, synthetic_project, metadata_csv):
        config, rows = synthetic_project
        bad = config.root / rows[0]['image_path']
        bad.write_bytes(b'not a PNG')  # Temporary file only.
        ds = HorseHumanDataset(metadata_csv(rows))
        with pytest.raises(ValueError, match='Cannot read image:') as error:
            ds[0]
        assert str(bad) in str(error.value)

    def test_d18_custom_transform_receives_unchanged_bgra(self, synthetic_project):
        config, _ = synthetic_project
        def transform(image):
            np.testing.assert_array_equal(image[0,0], [0,10,200,255])
            return torch.full((3,4,5), 0.75)
        spy = Mock(side_effect=transform)
        ds = HorseHumanDataset(config.splits_dir / 'train.csv', transform=spy)
        image, target = ds[0]
        spy.assert_called_once()
        assert torch.equal(image, torch.full((3,4,5), 0.75))
        assert target.item() == 0

    def test_d19_paths_independent_of_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ds = HorseHumanDataset(Path('data/splits/train.csv'))
        assert len(ds) == 845
        assert_image(ds[0][0], (3,64,64))

    def test_d20_lazy_loading(self, synthetic_project, monkeypatch):
        config, _ = synthetic_project
        reader = Mock(wraps=cv2.imread)
        monkeypatch.setattr(dataset_module.cv2, 'imread', reader)
        ds = HorseHumanDataset(config.splits_dir / 'train.csv')
        assert len(ds) == 6
        reader.assert_not_called()
        ds[0]
        reader.assert_called_once()
        assert reader.call_args.args[1] == cv2.IMREAD_UNCHANGED


class TestDataLoader:
    def test_l01_l11_all_real_images_once(self, monkeypatch):
        """One traversal covers shapes, sampler, batches, labels and exact paths."""
        reads = []
        read = cv2.imread
        def record(path, flags):
            reads.append(Path(path).resolve())
            return read(path, flags)
        monkeypatch.setattr(dataset_module.cv2, 'imread', record)
        loaders = build_loaders(CONFIG)
        assert set(loaders) == {'train','val','test'}
        assert reads == []
        total = 0
        for split, (count, batches, last) in {'train':(845,27,13),'val':(182,6,22),'test':(256,8,32)}.items():
            loader = loaders[split]
            assert isinstance(loader.sampler, RandomSampler if split == 'train' else SequentialSampler)
            assert loader.num_workers == 0 and not loader.drop_last
            assert len(loader) == batches
            metadata = pd.read_csv(CONFIG.splits_dir / f'{split}.csv')
            expected = {(CONFIG.root / r.image_path).resolve(): int(r.label) for r in metadata.itertuples()}
            start = len(reads)
            sizes = []
            for images, targets in loader:
                n = len(images)
                sizes.append(n)
                assert_image(images, (n,3,64,64))
                assert targets.shape == (n,1) and targets.dtype == torch.float32
                assert torch.isfinite(targets).all()
                assert set(targets.flatten().tolist()) <= {0,1}
                assert targets.flatten().tolist() == [expected[p] for p in reads[-n:]]
            assert sizes[0] == 32 and sizes[-1] == last
            assert sum(sizes) == count
            assert Counter(reads[start:]) == Counter(expected.keys())
            total += sum(sizes)
        assert total == len(reads) == len(set(reads)) == 1283

    def test_l12_l13_seeded_permutation(self, synthetic_project):
        config, _ = synthetic_project
        a = build_loaders(config)['train']
        b = build_loaders(config)['train']
        c = build_loaders(replace(config, seed=43))['train']
        order_a = list(a.sampler)
        assert order_a == list(b.sampler)
        assert sorted(order_a) == list(range(6))
        assert order_a != list(c.sampler)

    @pytest.mark.parametrize('split', ['val','test'])
    def test_l14_l15_eval_deterministic(self, synthetic_project, split):
        config, _ = synthetic_project
        loader = build_loaders(config)[split]
        first = list(loader)
        second = list(loader)
        assert len(first) == len(second)
        for (a,y), (b,z) in zip(first, second):
            assert torch.equal(a,b) and torch.equal(y,z)

    def test_l16_l17_custom_transform_routing(self, synthetic_project):
        config, _ = synthetic_project
        train = Mock(side_effect=lambda x: torch.full((3,4,5), .25))
        evaluate = Mock(side_effect=lambda x: torch.full((3,4,5), .75))
        loaders = build_loaders(config, train_transform=train, eval_transform=evaluate)
        assert train.call_count == evaluate.call_count == 0
        for split, value in [('train',.25),('val',.75),('test',.75)]:
            images, _ = next(iter(loaders[split]))
            assert images.shape == (2,3,4,5) and torch.all(images == value)
        assert train.call_count == 2
        assert evaluate.call_count == 4

    def test_l18_eval_does_not_advance_train_generator(self, synthetic_project):
        config, _ = synthetic_project
        a, b = build_loaders(config), build_loaders(config)
        state = a['train'].generator.get_state().clone()
        list(a['val']); list(a['test'])
        assert torch.equal(state, a['train'].generator.get_state())
        assert list(a['train'].sampler) == list(b['train'].sampler)

    def test_l12_actual_batch_order_reproducible(self, synthetic_project):
        config, _ = synthetic_project
        a, b = build_loaders(config)['train'], build_loaders(config)['train']
        # Six synthetic images encode unique values; compare actual iterator, not labels only.
        for (x,y),(z,t) in zip(a,b):
            assert torch.equal(x,z) and torch.equal(y,t)
