"""Restore hash-pinned R source packages into an isolated local runtime.

This does not renew historical binary qualification or approve any artifact.
Builds are serial with one native thread and one make job.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ORDER = ['Rcpp', 'jsonlite', 'data.table', 'iterators', 'foreach', 'shape',
         'R.methodsS3', 'R.oo', 'R.utils', 'R.matlab', 'RcppEigen',
         'LiblineaR', 'xgboost', 'glmnet']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def restore(directory):
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    archives = directory/'archives'
    library = directory/'library'
    logs = directory/'logs'
    for path in [archives, library, logs]:
        path.mkdir(exist_ok=True)
    pinned = json.loads((ROOT/'docs/reviews/andriy_r_runtime.json').read_text())
    files = {name.split('_', 1)[0]: (name, digest) for name, digest in pinned['package_archive_sha256'].items()}
    if set(files) != set(ORDER):
        raise ValueError('Pinned package closure changed')
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MAKEFLAGS='-j1',
               R_LIBS_USER=str(library), R_LIBS_SITE='', R_MAKEVARS_USER=str(directory/'Makevars'))
    (directory/'Makevars').write_text(pinned['makevars'])
    rscript = shutil.which('Rscript')
    r = shutil.which('R')
    if not rscript or not r:
        raise ValueError('Required R executable missing')
    version = subprocess.check_output([rscript, '--vanilla', '-e', 'cat(as.character(getRversion()))'], env=env, text=True).strip()
    if version != pinned['r_version']:
        raise ValueError('Recorded R version required')
    for package in ORDER:
        name, digest = files[package]
        target = archives/name
        if not target.exists():
            urls = [f'https://cran.r-project.org/src/contrib/{name}',
                    f'https://cran.r-project.org/src/contrib/Archive/{package}/{name}']
            for index, url in enumerate(urls):
                try:
                    with urllib.request.urlopen(url, timeout=60) as response:
                        data = response.read()
                except urllib.error.HTTPError as error:
                    if error.code == 404 and index == 0:
                        continue
                    raise
                if hashlib.sha256(data).hexdigest() != digest:
                    raise ValueError('Pinned archive hash differs: '+package)
                target.write_bytes(data)
                break
        if not target.is_file() or sha(target) != digest:
            raise ValueError('Missing or changed archive: '+package)
        print('Building '+package+' with one make job', flush=True)
        with (logs/(package+'.log')).open('w') as log:
            subprocess.run([r, 'CMD', 'INSTALL', '--no-multiarch', '--library='+str(library), str(target)],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    check = 'p <- c('+','.join(json.dumps(p) for p in ORDER)+'); for (n in p) { loadNamespace(n); cat(n, as.character(packageVersion(n)), "\\n") }'
    loaded = subprocess.check_output([rscript, '--vanilla', '-e', check], env=env, text=True)
    versions = dict(line.split() for line in loaded.splitlines())
    expected_versions = {package: name.split('_', 1)[1][:-7].replace('-', '.') for package, (name, _) in files.items()}
    if versions != expected_versions:
        raise ValueError('Restored package versions differ')
    libraries = {str(p.relative_to(library)): sha(p) for p in library.glob('*/libs/*.so')}
    old = json.loads((ROOT/'docs/reviews/andriy_r_contract_audit.json').read_text())['compiled_library_sha256']
    result = dict(restored=True, approved=False, catalog_mutations=0, native_threads=1, make_jobs=1,
                  r_version=version, loaded_packages=versions, compiled_library_sha256=libraries,
                  historical_binary_hashes_match=all(libraries.get(name) == digest for name, digest in old.items()),
                  requalification_required=True, restorer_sha256=sha(Path(__file__)))
    (directory/'restoration.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(restored=True, approved=False, historical_binary_hashes_match=result['historical_binary_hashes_match'], requalification_required=True)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    restore(parser.parse_args().directory)
