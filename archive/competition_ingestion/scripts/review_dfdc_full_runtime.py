"""Audit current DFDC numerical, augmentation, detector and native package integrity."""
import json
from pathlib import Path
from scripts.audit_dfdc_original_readiness import ROOT, sha
from scripts import review_video_numeric_runtime as checker

ROOT_PACKAGES=['numpy','torch','torchvision','timm','scipy','scikit-learn',
               'scikit-image','albumentations','opencv-python-headless','qudida','dlib','threadpoolctl']


def review():
    previous=checker.ROOT_PACKAGES
    checker.ROOT_PACKAGES=ROOT_PACKAGES
    try:
        runtime=checker.review()
    finally:
        checker.ROOT_PACKAGES=previous
    return dict(passed=True,approved=False,catalog_mutations=0,runtime=runtime,
        software_sha256={p:sha(ROOT/p) for p in ['scripts/review_dfdc_full_runtime.py',
            'sciona/dfdc_classifier.py','sciona/dfdc_detector.py','sciona/dfdc_hull_backend.py',
            'sciona/dfdc_transforms.py','sciona/dfdc_preparation.py']},
        limits=['Installed active base dependency RECORD integrity only; no model execution or publication approval.',
            'Native dlib package integrity does not validate caller-supplied predictor provenance, landmark semantics or detection quality.',
            'OpenCV uses the headless distribution; overlapping older opencv-python metadata is not qualified.',
            'Full detector, augmentation, B7 training and ensemble prediction must execute with explicit bounded resources.',
            'No historical binary equivalence, clean-install reproducibility or predictive accuracy claim.'])


if __name__=='__main__':
    report=review()
    (ROOT/'docs/reviews/dfdc_full_runtime.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,packages=len(report['runtime']['installed_packages']),approved=False)))
