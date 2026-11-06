"""Restore only the reviewed public CHAMPS software manifest, serially."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SHA256 = '99fde10617f64e20fc0459e73723ded08e3692518afd6e74bb67b8f96de804ca'


def verified(raw, entry):
    return (hashlib.sha256(raw).hexdigest() == entry['sha256'] and
            hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
            == entry['git_blob_sha1'])


def restore(destination):
    manifest = (ROOT / 'docs/reviews/competition_champs_source_pins.json').read_bytes()
    if hashlib.sha256(manifest).hexdigest() != MANIFEST_SHA256:
        raise ValueError('Reviewed software manifest changed')
    pins = json.loads(manifest)
    restored = 0
    with requests.Session() as session:
        for entry in pins['pins']:
            relative = Path(entry['software_path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe software manifest path')
            path = destination / relative
            if path.is_file() and verified(path.read_bytes(), entry):
                continue
            url = 'https://raw.githubusercontent.com/' + pins['repository'] + '/' + pins['commit'] + '/' + quote(relative.as_posix())
            response = session.get(url, timeout=30)
            response.raise_for_status()
            if not verified(response.content, entry):
                raise ValueError('Downloaded software differs from both reviewed hashes')
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(path.name + '.partial')
            partial.write_bytes(response.content)
            partial.replace(path)
            restored += 1
    return dict(passed=True, catalog_mutations=0, approved=False, software_only=True,
                manifest_sha256=MANIFEST_SHA256, source_commit=pins['commit'],
                verified_files=len(pins['pins']), restored_files=restored,
                scope='Verified public source code only; no model execution or runtime data download.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    result = restore(parser.parse_args().destination)
    (ROOT / 'docs/reviews/decomposed_champs_source_restore.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
