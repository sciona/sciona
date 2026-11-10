"""Restore reviewed public Cornell software only; no runtime inputs or weights."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tempfile
from urllib.request import urlopen
import zipfile

from sciona.cornell_augmentation import ROOT, PINS, DEPENDENCIES, manifest


def sha(data):
    return hashlib.sha256(data).hexdigest()


def relative(name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('Unsafe manifest path')
    return path


def verify(base, pins):
    names = [p['software_path'] for p in pins]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate manifest path')
    for pin in pins:
        data = (base / relative(pin['software_path'])).read_bytes()
        if sha(data) != pin['sha256']:
            raise ValueError('Restored software differs from reviewed manifest')
        if 'git_blob_sha1' in pin:
            blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            if blob != pin['git_blob_sha1']:
                raise ValueError('Restored Git blob differs')
    return len(pins)


def install(destination, pins, fetch):
    if destination.exists():
        return dict(verified_files=verify(destination, pins), downloaded_files=0)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='cornell-software-', dir=destination.parent) as temporary:
        staged = Path(temporary) / 'software'
        staged.mkdir()
        fetch(staged)
        count = verify(staged, pins)
        if destination.exists():
            raise ValueError('Destination appeared during restoration')
        staged.rename(destination)
    return dict(verified_files=count, downloaded_files=count)


def download(url):
    with urlopen(url, timeout=60) as response:
        return response.read()


def restore(source, dependencies):
    source_pins = manifest('competition_cornell_source_pins.json', PINS)
    dependency_pins = manifest('competition_cornell_historical_dependencies.json', DEPENDENCIES)
    if (source_pins['repository'], source_pins['commit']) != (
        'ryanwongsa/kaggle-birdsong-recognition', '4ad1aa4ed99bc097289c7593c55bc09234e0fc59'):
        raise ValueError('Unexpected reviewed source')

    def fetch_source(staged):
        for pin in source_pins['pins']:
            path = relative(pin['software_path'])
            data = download('https://raw.githubusercontent.com/' + source_pins['repository'] +
                            '/' + source_pins['commit'] + '/' + path.as_posix())
            if sha(data) != pin['sha256']:
                raise ValueError('Downloaded source hash differs')
            target = staged / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

    results = dict(source=install(source, source_pins['pins'], fetch_source), dependencies=[])
    print(json.dumps(dict(source=results['source'])), flush=True)
    for dep in dependency_pins:
        filename = dep['wheel']
        package, version = filename.split('-')[:2]

        def fetch_dependency(staged):
            index = json.loads(download('https://pypi.org/pypi/' + package + '/' + version + '/json'))
            choices = [r for r in index['urls'] if r['filename'] == filename and r['digests']['sha256'] == dep['sha256']]
            if len(choices) != 1:
                raise ValueError('Exact reviewed wheel unavailable')
            data = download(choices[0]['url'])
            if sha(data) != dep['sha256']:
                raise ValueError('Downloaded wheel hash differs')
            with zipfile.ZipFile(io.BytesIO(data)) as wheel:
                for pin in dep['pins']:
                    path = relative(pin['software_path'])
                    raw = wheel.read(path.as_posix())
                    if sha(raw) != pin['sha256']:
                        raise ValueError('Wheel member differs')
                    target = staged / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(raw)

        result = install(dependencies / package, dep['pins'], fetch_dependency)
        results['dependencies'].append(dict(wheel=filename, **result))
        print(json.dumps(results['dependencies'][-1]), flush=True)
    return dict(passed=True, catalog_mutations=0, training_launched=False,
                source_manifest_sha256=PINS, dependency_manifest_sha256=DEPENDENCIES,
                restorer_sha256=sha(Path(__file__).read_bytes()), **results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('/private/tmp/sciona_cornell_source'))
    parser.add_argument('--dependencies', type=Path, default=Path('/private/tmp/sciona_cornell_dependencies'))
    args = parser.parse_args()
    result = restore(args.source, args.dependencies)
    (ROOT / 'docs/reviews/cornell_reviewed_software_restored.json').write_text(json.dumps(result, indent=2)+'\n')
