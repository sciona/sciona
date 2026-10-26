"""Bind candidate link, import and notice evidence without approving a CDG."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def review():
    docs = ROOT / 'docs/reviews'
    paths = {kind: docs / ('competition_otto_candidate_' + kind + '.json')
             for kind in ('native_links', 'runtime_identity', 'native_notices')}
    records = {kind: json.loads(path.read_text()) for kind, path in paths.items()}
    links, runtime, notices = (records[k] for k in paths)
    for record in records.values():
        require(record['status'] == 'passed' and record['approved'] is False
                and record['catalog_mutations'] == 0, 'Candidate status differs')
    for record, script in ((links, 'validate_otto_candidate_native_links.py'),
                           (notices, 'retain_otto_candidate_native_notices.py')):
        require(record['verifier_sha256'] == sha(ROOT / 'scripts' / script), 'Verifier changed')
    inventory = docs / 'competition_otto_environment_inventory.json'
    require(links['inventory_sha256'] == sha(inventory), 'Inventory changed')
    require(links['historical_evidence_sha256'] == sha(docs / 'competition_otto_native_links.json'),
            'Historical evidence changed')
    required = {'scripts/validate_otto_candidate_runtime_identity.py',
                'docs/reviews/competition_otto_candidate_native_links.json',
                'docs/reviews/competition_otto_environment_inventory.json',
                'requirements/otto-execution.txt'}
    require(set(runtime['sha256']) == required, 'Runtime evidence coverage differs')
    for name, digest in runtime['sha256'].items():
        require(sha(ROOT / name) == digest, 'Runtime evidence changed')
    require(notices['native_link_evidence_sha256'] == sha(paths['native_links']), 'Notice closure differs')
    for name, record in links['native_files'].items():
        require(sha(Path(name)) == record['sha256'], 'Native binary changed')
        for link in record['links']:
            require(link.get('system') is True or link.get('target') in links['native_files'],
                    'Unrecorded dependency target')
    roots = set()
    for name in links['native_files']:
        parts = Path(name).parts
        if 'Cellar' in parts:
            i = parts.index('Cellar')
            roots.add(Path(*parts[:i+3]))
    require(set(notices['homebrew_packages']) == {p.parent.name + '/' + p.name for p in roots},
            'Native package notice coverage differs')
    for root in roots:
        package = notices['homebrew_packages'][root.parent.name + '/' + root.name]
        require(package['formula_sha256'] == sha(root / '.brew' / (root.parent.name + '.rb')),
                'Installed formula changed')
        require(bool(package['notice_origins']), 'Package lacks notices')
        require(all(origin in notices['license_origins'] for origin in package['notice_origins']),
                'Notice provenance missing')
    require(set(notices['license_origins'].values()) == set(notices['license_sha256']),
            'Retained notice coverage differs')
    for name, digest in notices['license_sha256'].items():
        require(sha(ROOT / 'docs/licenses' / name) == digest, 'Retained notice changed')
    require(runtime['verified_native_files'] == len(links['native_files']), 'Runtime closure differs')
    return dict(status='candidate_evidence_consistent', approved=False, catalog_mutations=0,
                native_files=len(links['native_files']), homebrew_packages=len(roots),
                distinct_native_notices=len(notices['license_sha256']),
                evidence_sha256={str(p.relative_to(ROOT)): sha(p) for p in paths.values()},
                reviewer_sha256=sha(Path(__file__)),
                remaining=['Qualify the decomposed CDG and every new reusable provider.',
                           'Execute the complete source stack with the current native closure.',
                           'Complete version-bound semantic, corruption, rollback and served-selection gates.'],
                limits=['Candidate links, imports and notices are consistent; this is not a new CDG approval.',
                        'No new model fits, empirical effectiveness, clean installation or dynamic-loading completeness claim.'])


if __name__ == '__main__':
    report = review()
    (ROOT / 'docs/reviews/competition_otto_candidate_runtime_review.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
