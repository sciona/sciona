"""Retain installed dependency notices without treating inventory as approval."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTICE_NAMES = {
    'licenses', 'license', 'license.txt', 'license.md', 'copying',
    'copying.txt', 'copying.md', 'notice', 'notice.txt', 'notice.md',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory():
    review_path = ROOT / 'docs/reviews/residual_classifier_binding_review.json'
    review = json.loads(review_path.read_text())
    if not review['passed'] or not review['dependency_closure']['compatible']:
        raise ValueError('Passing dependency preflight required')
    notices_root = ROOT / 'docs/licenses'
    existing = {}
    for path in sorted(notices_root.rglob('*')):
        if path.is_file():
            existing.setdefault(sha(path), path)
    packages, missing = {}, []
    for name, version in review['dependency_closure']['versions'].items():
        distribution = metadata.distribution(name)
        if distribution.version != version:
            raise ValueError('Dependency version drift: ' + name)
        retained = []
        for item in sorted(distribution.files or [], key=str):
            relative = Path(item)
            if not any(part.lower() in NOTICE_NAMES for part in relative.parts):
                continue
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unexpected notice location')
            source = Path(distribution.locate_file(item))
            if not source.is_file():
                continue
            digest = sha(source)
            target = existing.get(digest)
            if target is None:
                target = notices_root / 'residual-classifier' / name / relative
                if target.exists() and sha(target) != digest:
                    raise ValueError('Retained notice drift')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
                existing[digest] = target
            if sha(target) != digest:
                raise ValueError('Notice retention mismatch')
            retained.append(dict(installed_notice=str(relative),
                                 retained_notice=str(target.relative_to(ROOT)), sha256=digest))
        if not retained:
            missing.append(name)
        packages[name] = dict(
            version=version, retained_notices=retained,
            declared_license_expression=distribution.metadata.get('License-Expression'),
            declared_license=distribution.metadata.get('License'),
            declared_license_classifiers=[value for value in distribution.metadata.get_all('Classifier', [])
                                          if value.startswith('License ::')],
        )
    return dict(
        inventory_completed=True, approved=False, catalog_mutations=0,
        binding_review_sha256=sha(review_path), recorder_sha256=sha(Path(__file__)),
        packages=packages, packages_without_installed_notices=missing,
        limitations=[
            'Metadata is recorded verbatim as a software license declaration, not an approval or legal determination.',
            'Existing identical retained notices are referenced by content digest; their older directory labels do not establish provenance for this inventory.',
            'Missing installed notices require source evidence or an explicit reviewed scope disposition before the inventory can be considered complete.',
            'This scans installed distribution records, not an exhaustive native binary or dynamically loaded system-library inventory.',
        ],
    )


if __name__ == '__main__':
    report = inventory()
    (ROOT / 'docs/reviews/residual_classifier_dependency_notices.json').write_text(
        json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(packages=len(report['packages']),
        retained_notice_entries=sum(len(p['retained_notices']) for p in report['packages'].values()),
        missing=report['packages_without_installed_notices'], approved=False, catalog_mutations=0)))
