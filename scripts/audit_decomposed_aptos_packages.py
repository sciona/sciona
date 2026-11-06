"""Read-only installed RECORD verification for the candidate dependency set."""
import base64
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path


def main():
    audit_path = Path('docs/reviews/competition_aptos_environment_audit.json')
    audit = json.loads(audit_path.read_text())
    assert audit['dependency_constraints_satisfied']
    results = {}
    for name, selected in audit['packages'].items():
        distribution = metadata.distribution(name)
        assert distribution.version == selected['version']
        matched, unhashed, failures = 0, 0, []
        for entry in distribution.files or []:
            if not entry.hash:
                unhashed += 1
                continue
            path = Path(distribution.locate_file(entry))
            if not path.is_file():
                failures.append({'record_path': str(entry), 'reason': 'missing'})
                continue
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, entry.hash.mode).digest()
            encoded = base64.urlsafe_b64encode(digest).decode().rstrip('=')
            if encoded != entry.hash.value:
                failures.append({'record_path': str(entry), 'reason': 'hash mismatch'})
            else:
                matched += 1
        results[name] = {'version': distribution.version, 'matched_hashed_files': matched,
                         'unhashed_record_entries': unhashed, 'failures': failures}
        print(f'{name}: {matched} matched, {len(failures)} discrepancies', flush=True)
    report = {'approved': False, 'all_recorded_hashes_match': all(not item['failures'] for item in results.values()),
        'packages': results,
        'scope': 'Selected installed distribution files checked against their own RECORD. Does not authenticate RECORD against publisher wheels or verify every imported module origin; unrecorded/generated files are not covered.',
        'sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [
            str(audit_path), 'scripts/audit_decomposed_aptos_packages.py']}}
    Path('docs/reviews/decomposed_aptos_package_files.json').write_text(json.dumps(report, indent=2) + '\n')
    print('RECORD audit complete', flush=True)


if __name__ == '__main__':
    main()
