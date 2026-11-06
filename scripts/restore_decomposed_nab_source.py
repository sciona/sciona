"""Restore only pinned public benchmark software; never fetch benchmark data."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def restore(destination):
    pins = json.loads((ROOT/'docs/reviews/competition_nab_source_pins.json').read_text())
    if pins['repository'] != 'numenta/NAB' or pins['commit'] != 'ea702d75cc2258d9d7dd35ca8e5e2539d71f3140':
        raise ValueError('Unreviewed source identity')
    restored = 0
    for entry in pins['pins']:
        relative = Path(entry['software_path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe software path')
        path = destination/relative
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']:
            continue
        url = f"https://raw.githubusercontent.com/{pins['repository']}/{pins['commit']}/{relative.as_posix()}"
        with urlopen(url, timeout=60) as response:
            raw = response.read()
        if hashlib.sha256(raw).hexdigest() != entry['sha256']:
            raise ValueError('Public software pin mismatch')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        restored += 1
    return dict(verified_files=len(pins['pins']), restored_files=restored,
                source_commit=pins['commit'], software_only=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    print(json.dumps(restore(parser.parse_args().destination)))
