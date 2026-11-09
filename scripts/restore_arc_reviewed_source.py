"""Restore only manifest-pinned public solver software, serially and hash checked."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.request import urlopen

from sciona.arc_build import ROOT, PINS_SHA256, digest, verified_sources


def restore(destination: Path):
    raw = (ROOT / 'docs/reviews/competition_arc_source_pins.json').read_bytes()
    if digest(raw) != PINS_SHA256:
        raise ValueError('Reviewed source manifest changed')
    manifest = json.loads(raw)
    if destination.exists():
        return dict(verified_files=len(verified_sources(destination)), downloaded_files=0)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='arc-source-restore-', dir=destination.parent) as temporary:
        staged = Path(temporary) / 'source'
        staged.mkdir()
        seen = set()
        for entry in manifest['files']:
            relative = Path(entry['path'])
            if relative.is_absolute() or '..' in relative.parts or relative.as_posix() in seen:
                raise ValueError('Invalid or repeated manifest path')
            seen.add(relative.as_posix())
            url = 'https://raw.githubusercontent.com/top-quarks/ARC-solution/' + manifest['commit'] + '/' + relative.as_posix()
            with urlopen(url, timeout=30) as response:
                data = response.read()
            blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            if digest(data) != entry['sha256'] or blob != entry['git_blob_sha1']:
                raise ValueError('Downloaded source differs from reviewed manifest')
            target = staged / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        verified = verified_sources(staged)
        if destination.exists():
            raise ValueError('Destination appeared during restoration')
        staged.rename(destination)
    return dict(verified_files=len(verified), downloaded_files=len(verified))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    print(json.dumps(restore(parser.parse_args().destination)))
