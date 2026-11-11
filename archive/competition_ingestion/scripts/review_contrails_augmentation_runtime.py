"""Supplement Contrails numerical integrity with the lazy augmentation closure."""
import json
from pathlib import Path
from scripts.audit_contrails_original_readiness import ROOT, sha
from scripts import review_video_numeric_runtime as checker


def review():
    previous = checker.ROOT_PACKAGES
    checker.ROOT_PACKAGES = ['albumentations', 'opencv-python-headless', 'qudida']
    try:
        runtime = checker.review()
    finally:
        checker.ROOT_PACKAGES = previous
    old = json.loads((ROOT/'docs/reviews/competition_contrails_preparation_environment.json').read_text())
    return dict(passed=True, approved=False, catalog_mutations=0, runtime=runtime,
        software_sha256={p:sha(ROOT/p) for p in ['sciona/contrails_preparation.py',
            'scripts/review_contrails_augmentation_runtime.py',
            'docs/reviews/competition_contrails_preparation_environment.json']},
        historical_version_comparison=[dict(package=name, historical=version,
            current=runtime['installed_packages'][name]['version'])
            for name,version in old['versions'].items() if name != 'torchvision'],
        limits=['Supplemental RECORD verification of lazy augmentation dependencies; no model execution or approval.',
                'This check was added after the current v47 run started, so it does not establish augmentation integrity before that run.',
                'Corrected stored-graph qualification must verify both numerical and augmentation closures before and after execution.'])


if __name__ == '__main__':
    report = review()
    (ROOT/'docs/reviews/contrails_augmentation_runtime.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True, packages=len(report['runtime']['installed_packages']), approved=False)))
