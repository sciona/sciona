"""Audit overlapping OpenCV installations and test a RECORD-only private copy.

Does not install, uninstall, or modify the shared Python environment. Independent
wheel authentication and historical OpenCV equivalence remain separate gates.
"""
import argparse
import base64
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]


def digest(path, algorithm='sha256'):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, algorithm).digest()


def audit_distribution(name):
    dist = metadata.distribution(name)
    matched, missing, mismatched, unhashed = [], [], [], []
    for entry in dist.files or []:
        if entry.hash is None:
            unhashed.append(str(entry))
            continue
        path = Path(dist.locate_file(entry))
        if not path.is_file():
            missing.append(str(entry))
        elif base64.urlsafe_b64encode(digest(path, entry.hash.mode)).rstrip(b'=').decode() != entry.hash.value:
            mismatched.append(str(entry))
        else:
            matched.append(str(entry))
    active = []
    for raw in dist.requires or []:
        req = Requirement(raw)
        if req.marker and not req.marker.evaluate({'extra': ''}):
            continue
        if req.url or req.extras or not req.specifier.contains(metadata.version(req.name), prereleases=True):
            raise ValueError('Unsatisfied or unreviewed OpenCV requirement')
        active.append(str(req))
    return dict(version=dist.version, matched_hashed_files=len(matched),
                missing=missing, mismatched=mismatched, unhashed_entries=len(unhashed),
                active_requirements=active)


CHILD = '''
import json, sys
from pathlib import Path
import cv2
overlay = Path(sys.argv[1]).resolve()
cv2.setNumThreads(0)
if cv2.getNumThreads() != 1:
    raise ValueError('OpenCV serial execution required')
if not Path(cv2.__file__).resolve().is_relative_to(overlay):
    raise ValueError('OpenCV Python module escaped isolated namespace')
native = Path(cv2._native.__file__).resolve()
if not native.is_relative_to(overlay):
    raise ValueError('OpenCV native module escaped isolated namespace')
from scripts.validate_osic_image_preprocessing import validate
result = validate(sys.argv[2])
modules = [str(native.relative_to(overlay))]
for name, module in tuple(sys.modules.items()):
    if name == 'cv2' or name.startswith('cv2.'):
        source = getattr(module, '__file__', None)
        if source:
            path = Path(source).resolve()
            if not path.is_relative_to(overlay):
                raise ValueError('OpenCV module escaped isolated namespace')
            modules.append(str(path.relative_to(overlay)))
print(json.dumps(dict(runtime=cv2.__version__, imported_files=sorted(set(modules)),
    cases=result['cases'], source_function_sha256=result['source_function_sha256'],
    software_sha256=result['software_sha256'], serial_execution=True)))
'''


def audit(source_directory):
    names = ('opencv-python', 'opencv-python-headless')
    before = {name: audit_distribution(name) for name in names}
    selected = metadata.distribution('opencv-python-headless')
    if (before['opencv-python-headless']['missing'] or before['opencv-python-headless']['mismatched']
            or not before['opencv-python-headless']['matched_hashed_files']):
        raise ValueError('Cannot isolate a distribution with mismatched RECORD files')
    copied = {}
    with tempfile.TemporaryDirectory(prefix='sciona-osic-opencv-') as temporary:
        overlay = Path(temporary)
        for entry in selected.files or []:
            relative = Path(str(entry))
            if not entry.hash or relative.parts[0] != 'cv2':
                continue
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe installed package path')
            target = overlay/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(selected.locate_file(entry), target)
            if base64.urlsafe_b64encode(digest(target, entry.hash.mode)).rstrip(b'=').decode() != entry.hash.value:
                raise ValueError('Copied package changed')
            copied[str(relative)] = digest(target).hex()
        if 'cv2/__init__.py' not in copied or 'cv2/cv2.abi3.so' not in copied:
            raise ValueError('Complete Python and native module required')
        env = dict(os.environ, PYTHONPATH=str(overlay)+os.pathsep+str(ROOT),
            PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
            MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        completed = subprocess.run([str(ROOT/'.venv/bin/python'), '-B', '-c', CHILD,
            str(overlay), str(Path(source_directory).resolve())], cwd=ROOT, env=env,
            capture_output=True, text=True, timeout=120, check=True)
        result = json.loads(completed.stdout)
        if copied != {relative: digest(overlay/relative).hex() for relative in copied}:
            raise ValueError('Isolated package changed during execution')
        actual = {str(p.relative_to(overlay)) for p in overlay.rglob('*') if p.is_file()}
        if actual != set(copied):
            raise ValueError('Isolated namespace acquired unrecorded files')
    if before != {name: audit_distribution(name) for name in names}:
        raise ValueError('Shared environment changed during audit')
    reference = json.loads((ROOT/'docs/reviews/osic_image_preprocessing_validation.json').read_text())
    for field in ('cases', 'source_function_sha256', 'software_sha256'):
        if result[field] != reference[field]:
            raise ValueError('Isolated execution differs from qualified source comparisons')
    return dict(format='osic-opencv-isolation.v1', passed=True, approved=False,
        catalog_mutations=0, shared_environment_mutations=0, installed_distributions=before,
        isolated_distribution='opencv-python-headless', isolated_cv2_files=len(copied),
        isolated_file_inventory_sha256=hashlib.sha256(json.dumps(copied, sort_keys=True).encode()).hexdigest(),
        runtime=result['runtime'], imported_files=result['imported_files'],
        synthetic_cases=len(result['cases']), values_compared=sum(r['values_compared'] for r in result['cases']),
        exact_previous_source_comparison=True, no_unrecorded_overlay_files=True,
        serial_execution=result['serial_execution'],
        reference_sha256=digest(ROOT/'docs/reviews/osic_image_preprocessing_validation.json').hex(),
        auditor_sha256=digest(Path(__file__)).hex(),
        limitations=['Only the OpenCV namespace was isolated; other dependencies still come from the shared virtual environment.',
            'Installed package RECORD is not independent publisher authentication.',
            'The shared installation remains conflicted and is not approved by this experiment.',
            'This establishes current-backend preprocessing equality without leftover OpenCV 4 files; historical backend, decoding and training remain unqualified.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory', required=True)
    result = audit(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/osic_opencv_isolation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, isolated_cv2_files=result['isolated_cv2_files'],
        synthetic_cases=result['synthetic_cases'], shared_environment_mutations=0)))
