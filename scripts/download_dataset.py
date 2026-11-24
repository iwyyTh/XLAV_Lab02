"""Download official archives, validate bytes/images, retain source split provenance.

Input: optional --archive-dir containing the two original ZIP files.
Output: data/raw/{horses,humans} and data/source_manifest.csv. Never overwrites images.
"""
import argparse
import csv
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'train': 'https://storage.googleapis.com/download.tensorflow.org/data/horse-or-human.zip',
    'test': 'https://storage.googleapis.com/download.tensorflow.org/data/validation-horse-or-human.zip',
}
EXPECTED = {'train': 1027, 'test': 256}


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-dir', type=Path, default=ROOT / 'data/downloads')
    args = parser.parse_args()
    manifest = ROOT / 'data/source_manifest.csv'
    if manifest.exists():
        with manifest.open(newline='') as f:
            rows = list(csv.DictReader(f))
        assert len({r['image_path'] for r in rows}) == 1283, 'Incomplete manifest'
        for split, count in EXPECTED.items():
            assert sum(r['official_split'] == split for r in rows) == count
        for row in rows:
            path = ROOT / row['image_path']
            assert path.resolve().is_relative_to((ROOT / 'data/raw').resolve())
            assert sha256(path) == row['sha256'], f'Changed image: {path}'
            with Image.open(path) as im:
                im.load()
        print('Existing official dataset validated; download skipped.')
        return
    args.archive_dir.mkdir(parents=True, exist_ok=True)
    records = []
    # Validate both ZIPs before publishing source provenance.
    for split, url in SOURCES.items():
        archive = args.archive_dir / url.rsplit('/', 1)[1]
        if not archive.exists():
            temporary = archive.with_suffix('.zip.part')
            print(f'Downloading {url}', flush=True)
            with urllib.request.urlopen(url, timeout=30) as response, temporary.open('wb') as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            temporary.replace(archive)
        with zipfile.ZipFile(archive) as zf:
            members = sorted(n for n in zf.namelist()
                             if len(Path(n).parts) == 2
                             and Path(n).parts[0] in ('horses', 'humans')
                             and Path(n).suffix.lower() == '.png')
            assert len(members) == EXPECTED[split], f'Unexpected image count: {archive}'
            for name in members:
                content = zf.read(name)  # ZIP CRC is checked while reading.
                with Image.open(io.BytesIO(content)) as im:
                    im.verify()
                with Image.open(io.BytesIO(content)) as im:
                    im.load()
                cls = Path(name).parts[0]
                # Prefix preserves original train/test membership and avoids collisions.
                dest = ROOT / 'data/raw' / cls / f'{split}__{Path(name).name}'
                dest.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256(content).hexdigest()
                if dest.exists():
                    assert sha256(dest) == digest, f'Refusing to overwrite {dest}'
                else:
                    dest.write_bytes(content)
                records.append(dict(image_path=dest.relative_to(ROOT).as_posix(),
                                    label=0 if cls == 'horses' else 1, class_name=cls,
                                    official_split=split, sha256=digest, source_url=url))
    with manifest.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f'Validated and organized {len(records)} images.')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'Download failed: {type(exc).__name__}: {exc}', file=sys.stderr)
        print('Manual download: put both original ZIPs in data/downloads and rerun.', file=sys.stderr)
        for url in SOURCES.values():
            print(url, file=sys.stderr)
        sys.exit(1)
