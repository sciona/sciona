"""Check OSIC's installed numerical runtime closure without importing models.

The three pinned overlay distributions are reviewed separately by
review_osic_archive_notices.py. Their active installed dependencies are roots
here. This check does not mutate the environment used by live training jobs.
"""
import hashlib
import json
from pathlib import Path

from scripts import review_video_numeric_runtime as checker

ROOT = Path(__file__).resolve().parents[1]
ROOT_PACKAGES = [
    'tensorflow', 'tf-keras', 'numpy', 'pandas', 'scipy', 'scikit-learn',
    'scikit-image', 'h5py', 'opencv-python-headless', 'pydantic', 'icontract',
]


def review():
    previous = checker.ROOT_PACKAGES
    checker.ROOT_PACKAGES = ROOT_PACKAGES
    try:
        runtime = checker.review()
    finally:
        checker.ROOT_PACKAGES = previous
    return dict(
        format='osic-runtime-integrity.v1', passed=True, approved=False,
        catalog_mutations=0, package_installations=0,
        runtime=runtime,
        software_sha256={
            name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in ('scripts/review_osic_runtime_integrity.py',
                         'scripts/review_video_numeric_runtime.py')
        },
        limits=[
            'Current-platform active base requirements and installed RECORD integrity only.',
            'Pinned efficientnet, keras-applications and pydicom overlays are reviewed separately; they are not resolved from installed distributions here.',
            'OpenCV qualification requires the verified headless overlay; overlapping opencv-python metadata is not qualified by this review.',
            'No native-library license review, publisher authentication or clean-install reproducibility claim.',
            'No model execution, original workflow completion, provider registration or publication approval.',
        ],
    )


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/osic_runtime_integrity.json').write_text(
        json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, approved=False,
                         packages=len(result['runtime']['installed_packages']))))
