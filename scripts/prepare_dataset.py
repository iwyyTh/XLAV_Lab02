"""Audit real images and freeze stratified CSV splits (seed 42).

Preserve official test if a verified source manifest exists; otherwise use 70/15/15.
Refuse corrupt images, label conflicts and changed existing splits. Exact duplicates
are excluded deterministically (official test has priority); original files stay intact.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
LABELS = {'horses': 0, 'humans': 1}
FIELDS = ['image_path', 'label', 'class_name']


def audit(root: Path = ROOT) -> tuple[pd.DataFrame, list[dict]]:
    """Read every raw file; return metadata and unreadable-file diagnostics."""
    rows, errors = [], []
    for cls, label in LABELS.items():
        for path in sorted((root / 'data/raw' / cls).rglob('*')):
            if not path.is_file() or path.name.startswith('.'):
                continue
            try:
                with Image.open(path) as im:
                    im.verify()
                with Image.open(path) as im:
                    im.load()
                    row = dict(image_path=path.relative_to(root).as_posix(), label=label,
                               class_name=cls, width=im.width, height=im.height,
                               format=im.format, mode=im.mode, channels=len(im.getbands()))
                with path.open('rb') as stream:
                    row['sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
                rows.append(row)
            except Exception as exc:
                errors.append({'image_path': path.relative_to(root).as_posix(), 'error': str(exc)})
    return pd.DataFrame(rows), errors


def prepare(root: Path = ROOT, check_only: bool = False) -> dict:
    """Input: project root. Output: audit summary; create splits only on first run."""
    df, errors = audit(root)
    metrics = root / 'results/metrics'
    metrics.mkdir(parents=True, exist_ok=True)
    report = {'readable_images': len(df), 'corrupt_images': errors, 'status': 'FAIL'}
    report_path = metrics / 'dataset_audit.json'
    report_path.write_text(json.dumps(report, indent=2))
    if errors or df.empty or set(df.class_name) != set(LABELS):
        raise ValueError('Missing classes/dataset or unreadable files; see dataset_audit.json')
    df.to_csv(metrics / 'image_inventory.csv', index=False)
    if df.groupby('sha256').label.nunique().max() > 1:
        raise ValueError('Identical bytes have conflicting class labels')
    manifest = root / 'data/source_manifest.csv'
    if manifest.exists():
        source = pd.read_csv(manifest)
        assert source.image_path.is_unique
        assert set(source.image_path) == set(df.image_path), 'Raw files differ from source manifest'
        merged = df.merge(source[['image_path', 'sha256', 'official_split']], on='image_path', suffixes=('', '_source'))
        assert (merged.sha256 == merged.sha256_source).all(), 'Source hashes changed'
        assert set(merged.official_split) == {'train', 'test'}
        assert (merged.official_split == 'test').sum() == 256
        assert (merged.official_split == 'train').sum() == 1027
        df = merged
        # Alphabetic test precedes train, retaining official holdout on duplicate hashes.
        df = df.sort_values(['official_split', 'image_path'])
    else:
        df = df.sort_values('image_path')
    excluded = df[df.duplicated('sha256', keep='first')]
    unique = df.drop_duplicates('sha256', keep='first').copy()
    if manifest.exists():
        test = unique[unique.official_split == 'test']
        if len(test) != 256:
            raise ValueError('Duplicates within official test; manual review needed before changing holdout')
        pool = unique[unique.official_split == 'train']
        train, val = train_test_split(pool, test_size=15/85, stratify=pool.label, random_state=42)
        policy = 'Official TFDS test retained; remaining train:val = 70:15'
    else:
        train, rest = train_test_split(unique, test_size=.30, stratify=unique.label, random_state=42)
        val, test = train_test_split(rest, test_size=.50, stratify=rest.label, random_state=42)
        policy = 'No official provenance; stratified 70/15/15'
    splits = dict(train=train, val=val, test=test)
    assert set.union(*(set(x.sha256) for x in splits.values())) == set(unique.sha256)
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        assert set(splits[a].sha256).isdisjoint(splits[b].sha256)
        assert set(splits[a].image_path).isdisjoint(splits[b].image_path)
    for part in splits.values():
        assert set(part.label) == {0, 1}
        for row in part.itertuples():
            assert not Path(row.image_path).is_absolute()
            assert (root / row.image_path).is_file()
            assert LABELS[row.class_name] == row.label
    destination = root / 'data/splits'
    destination.mkdir(parents=True, exist_ok=True)
    texts = {name: part.sort_values('image_path')[FIELDS].to_csv(index=False) for name, part in splits.items()}
    # Validate all old files before writing any new one. Header-only placeholders are allowed.
    for name, content in texts.items():
        target = destination / f'{name}.csv'
        if target.exists() and len(pd.read_csv(target)):
            assert target.read_text() == content, f'Refusing to change frozen split: {target}'
        elif check_only:
            raise ValueError(f'Missing populated split: {target}')
    fingerprint = hashlib.sha256(unique.sort_values('image_path')[['image_path','sha256']].to_csv(index=False).encode()).hexdigest()
    lock = destination / 'split_metadata.json'
    if lock.exists():
        assert json.loads(lock.read_text())['dataset_sha256'] == fingerprint, 'Dataset content changed'
    elif check_only:
        raise ValueError('Missing split metadata')
    summary = {name: {cls: int((part.class_name == cls).sum()) for cls in LABELS} for name, part in splits.items()}
    metadata = dict(seed=42, policy=policy, dataset_sha256=fingerprint, counts=summary,
                    ratios={k: len(v)/len(unique) for k,v in splits.items()},
                    excluded_duplicates=excluded[['image_path','sha256']].to_dict('records'))
    if not check_only:
        for name, content in texts.items():
            (destination / f'{name}.csv').write_text(content)
        lock.write_text(json.dumps(metadata, indent=2))
    report.update(status='PASS', unique_images=len(unique), duplicate_files=len(excluded),
                  counts=summary, policy=policy, ratios=metadata['ratios'],
                  sizes=df.groupby(['width','height']).size().to_dict().__str__(),
                  formats=df['format'].value_counts().to_dict(), modes=df['mode'].value_counts().to_dict(),
                  channels=df.channels.value_counts().to_dict())
    report_path.write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(check_only=args.check_only), indent=2))
    except Exception as exc:
        print(f'Preparation failed: {type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(1)
