"""Read-only dependency resolution audit for the combined reference runtime."""
import base64
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


DECLARED_ROOTS = ['torch', 'torchvision', 'timm', 'tensorflow', 'tf-keras', 'numpy',
         'albumentations', 'opencv-python-headless', 'scipy', 'scikit-image', 'safetensors', 'h5py']
OBSERVED_IMPORT_ROOTS = ['python-dateutil', 'h2', 'hpack', 'hyperframe', 'jax', 'jaxlib',
                         'numexpr', 'pandas', 'pyarrow', 'pytz', 'sniffio', 'zstandard']
ROOTS = DECLARED_ROOTS + OBSERVED_IMPORT_ROOTS


def main():
    pending = [(name, frozenset()) for name in ROOTS]
    seen, packages, failures = set(), {}, []
    while pending:
        name, extras = pending.pop()
        name = canonicalize_name(name)
        state = (name, extras)
        if state in seen:
            continue
        seen.add(state)
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            failures.append({'package': name, 'reason': 'missing'})
            continue
        selected = packages.setdefault(name, {'version': distribution.version, 'active_requirements': []})
        for value in distribution.requires or []:
            requirement = Requirement(value)
            if requirement.marker and not any(requirement.marker.evaluate({'extra': extra}) for extra in ({''} | set(extras))):
                continue
            if value not in selected['active_requirements']:
                selected['active_requirements'].append(value)
            try:
                installed = metadata.version(requirement.name)
            except metadata.PackageNotFoundError:
                failures.append({'package': name, 'requirement': value, 'reason': 'missing'})
                continue
            if installed not in requirement.specifier:
                failures.append({'package': name, 'requirement': value, 'installed': installed, 'reason': 'version mismatch'})
            pending.append((requirement.name, frozenset(requirement.extras)))
    # Both GUI and headless distributions are present in the broad workspace.
    # Determine which RECORD actually matches the imported binary.
    import cv2
    binary = Path(cv2.__file__).parent / 'cv2.abi3.so'
    data = binary.read_bytes()
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('=')
    providers = {}
    for name in ['opencv-python', 'opencv-python-headless']:
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        record = next((item for item in dist.files or [] if str(item) == 'cv2/cv2.abi3.so'), None)
        providers[name] = {'version': dist.version, 'binary_record_matches': bool(record and record.hash and record.hash.mode == 'sha256' and record.hash.value == digest)}
    headless_matches = providers.get('opencv-python-headless', {}).get('binary_record_matches', False)
    report = {'approved': False, 'dependency_constraints_satisfied': not failures,
        'roots': ROOTS, 'declared_roots': DECLARED_ROOTS, 'observed_import_roots': OBSERVED_IMPORT_ROOTS,
        'python_version': sys.version.split()[0], 'packages': dict(sorted(packages.items())),
        'failures': failures, 'opencv_binary_providers': providers,
        'declared_headless_binary_matches': headless_matches,
        'scope': 'Installed combined reference dependency constraints and OpenCV binary attribution only; complete file origins, reproducible installation and license notices remain separate gates.',
        'sha256': {'scripts/audit_decomposed_cassava_environment.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    Path('docs/reviews/decomposed_cassava_environment.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'packages': len(packages), 'failures': failures, 'opencv_binary_providers': providers}))


if __name__ == '__main__':
    main()
