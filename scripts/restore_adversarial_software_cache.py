"""Restore only the immutable public software files listed in reviewed manifests."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def restore():
    reviews = ROOT/'docs/reviews'
    pins = json.loads((reviews/'competition_adversarial_source_pins.json').read_text())
    sources = [(s, Path('/private/tmp/sciona_adversarial_source')/s['branch']) for s in pins['sources']]
    for name, directory in [('tensorflow', 'sciona_adversarial_tensorflow_source'),
                            ('pooling', 'sciona_adversarial_tensorflow_source'),
                            ('scipy', 'sciona_adversarial_scipy_source')]:
        sources.append((json.loads((reviews/f'competition_adversarial_{name}_reference.json').read_text()),
                        Path('/private/tmp')/directory))
    checked = downloaded = 0
    for source, destination in sources:
        commit = source['commit']
        if len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):
            raise ValueError('Immutable commit required')
        repo = source['repository']
        if not repo.startswith('https://github.com/'):
            raise ValueError('Reviewed GitHub software repository required')
        base = repo.replace('https://github.com/', 'https://raw.githubusercontent.com/')+'/'+commit
        for item in source['files']:
            relative = Path(item['path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe source path')
            path = destination/relative
            data = path.read_bytes() if path.exists() else None
            if data is None or hashlib.sha256(data).hexdigest()!=item['sha256']:
                request = urllib.request.Request(base+'/'+item['path'], headers={'User-Agent':'sciona-source-review'})
                with urllib.request.urlopen(request, timeout=30) as response:
                    data = response.read()
                if hashlib.sha256(data).hexdigest()!=item['sha256']:
                    raise ValueError('Downloaded software hash mismatch')
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                downloaded += 1
            if 'git_blob_sha1' in item:
                actual = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
                if actual!=item['git_blob_sha1']:
                    raise ValueError('Git software object hash mismatch')
            checked += 1
    return dict(software_files_verified=checked, software_files_downloaded=downloaded,
                data_or_model_downloads=0, workers=1)


if __name__ == '__main__':
    print(json.dumps(restore()))
