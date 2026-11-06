"""Restore only previously reviewed public model software into a private cache."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

import requests

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def download(url, path, digest, size=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and sha(path) == digest and (size is None or path.stat().st_size == size):
        return
    partial = path.with_suffix(path.suffix+'.partial')
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with partial.open('wb') as stream:
            for chunk in response.iter_content(1024*1024):
                stream.write(chunk)
    if sha(partial) != digest or (size is not None and partial.stat().st_size != size):
        raise ValueError('Public model artifact differs from reviewed pin')
    partial.replace(path)


def restore(cache_root):
    reviews = ROOT/'docs/reviews'
    paths = [reviews/('competition_cassava_'+suffix+'.json') for suffix in
             ('timm_pretrained_validation', 'efficientnet_weight_origin_validation', 'cropnet_artifact')]
    timm, efficientnet, cropnet = [json.loads(p.read_text()) for p in paths]
    verified = []
    def report(unresolved):
        return dict(passed=not unresolved, catalog_mutations=0, approved=False, model_software_only=True,
            expected_artifacts=4, verified_artifacts=len(verified), artifacts=verified, unresolved=unresolved,
            sha256={str(p.relative_to(ROOT)): sha(p) for p in [*paths, Path(__file__).resolve()]},
            limitations=['Only public model software is downloaded; no private runtime images or training records are accessed.',
                'Previously reviewed artifact hashes are preserved. This establishes integrity, not new use or redistribution rights.',
                'All four artifacts are required. A partial restore cannot qualify the complete computation.',
                'Backbone loading, full training, runtime qualification and catalog publication remain separate checks.'])
    for tag, row in timm['references'].items():
        path = cache_root/'sciona_cassava_timm_pretrained'/(tag+'.safetensors')
        download(row['weights_url'], path, row['weights_sha256'], row['weights_bytes'])
        verified.append(dict(family=tag, sha256=sha(path)))
        print('Verified public backbone:', tag, flush=True)
    row = efficientnet['upstream_artifact']
    directory = cache_root/'sciona_cassava_efficientnet_upstream_reference'
    path = directory/'model.h5'
    download(row['url'], path, row['sha256'], row['bytes'])
    (directory/'manifest.json').write_text(json.dumps(row, indent=2)+'\n')
    verified.append(dict(family='efficientnet', sha256=sha(path)))
    print('Verified public EfficientNet artifact', flush=True)
    directory = cache_root/'sciona_cassava_cropnet_reference'
    archive = directory/'model.tar.gz'
    try:
        download(cropnet['reference'], archive, cropnet['archive_sha256'])
    except requests.HTTPError as error:
        return report([dict(family='cropnet', reason='Reviewed artifact endpoint returned HTTP error',
            http_status=error.response.status_code)])
    model = directory/'model'
    expected = cropnet['files']
    with tarfile.open(archive) as tar:
        members = {m.name.removeprefix('./'): m for m in tar.getmembers() if m.isfile()}
        if set(members) != set(expected):
            raise ValueError('Public CropNet archive file set differs from reviewed manifest')
        for name, digest in expected.items():
            target = model/name
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('Unsafe reviewed artifact member path')
            with tar.extractfile(members[name]) as stream:
                data = stream.read()
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError('Public CropNet member hash mismatch')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    actual = {p.relative_to(model).as_posix(): sha(p) for p in model.rglob('*') if p.is_file()}
    if actual != expected or any(p.is_symlink() for p in model.rglob('*')):
        raise ValueError('Cached CropNet tree differs')
    verified.append(dict(family='cropnet', archive_sha256=sha(archive), files_verified=len(expected)))
    print('Verified public CropNet artifact', flush=True)
    return report([])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root', type=Path, required=True)
    report = restore(parser.parse_args().cache_root)
    (ROOT/'docs/reviews/decomposed_cassava_artifact_restore.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('passed', 'verified_artifacts', 'unresolved')}))
    if not report['passed']:
        raise SystemExit(1)
