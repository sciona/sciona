"""Read-only comparison of retained native evidence with installed software.

Does not repair libraries, replace retained evidence, load models or approve
providers. Homebrew opt aliases are candidates, never accepted substitutes.
Only software-installation metadata is written; no runtime inputs are read.
"""
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


def audit():
    source = ROOT / 'docs/reviews/competition_otto_native_links.json'
    retained = json.loads(source.read_text())
    unchanged = 0
    differences = []
    for name, record in sorted(retained['native_files'].items()):
        path = Path(name)
        actual = sha(path) if path.is_file() else None
        if actual == record['sha256']:
            unchanged += 1
            continue
        item = dict(software_path=name, recorded_sha256=record['sha256'],
                    status='missing' if actual is None else 'changed',
                    installed_sha256=actual)
        cellar = Path('/opt/homebrew/Cellar')
        if path.is_relative_to(cellar):
            parts = path.relative_to(cellar).parts
            candidate = Path('/opt/homebrew/opt') / parts[0] / Path(*parts[2:])
            if candidate.is_file():
                candidate_hash = sha(candidate)
                item['unqualified_opt_candidate'] = dict(
                    software_path=str(candidate.resolve()), sha256=candidate_hash,
                    matches_retained_bytes=candidate_hash == record['sha256'])
        differences.append(item)
    return dict(status='drift_detected' if differences else 'retained_bytes_match',
                read_only=True, catalog_mutations=0, approved=False,
                retained_native_files=len(retained['native_files']), unchanged=unchanged,
                missing=sum(x['status'] == 'missing' for x in differences),
                changed=sum(x['status'] == 'changed' for x in differences),
                differences=differences, retained_evidence_sha256=sha(source),
                auditor_sha256=sha(Path(__file__)),
                limits=['File-byte comparison only; no native loading or fit execution.',
                        'Opt candidates require new dependency/link/runtime qualification.',
                        'Historical evidence and served implementation bytes are unchanged.'])


if __name__ == '__main__':
    report = audit()
    (ROOT / 'docs/reviews/competition_otto_native_drift.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('differences', 'limits')}))
