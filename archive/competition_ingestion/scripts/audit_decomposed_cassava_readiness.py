"""Record current prerequisites without rewriting historical Cassava evidence."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path

from scripts.import_flavours_provider_drafts import ROOT, connect


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def audit():
    reviews = ROOT/'docs/reviews'
    environment_path = reviews/'competition_cassava_environment_audit.json'
    environment = json.loads(environment_path.read_text())
    package_mismatches = []
    for name, selected in environment['packages'].items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual = None
        if actual != selected['version']:
            package_mismatches.append(dict(package=name, expected=selected['version'], actual=actual))
    reports = {}
    evidence_paths = [environment_path, Path(__file__).resolve()]
    for suffix in ('pipeline_validation', 'graph_execution_validation'):
        path = reviews/('competition_cassava_'+suffix+'.json')
        report = json.loads(path.read_text())
        reports[suffix] = dict(historical_passed=report['passed'], changed_implementation_files=[
            name for name, digest in report['sha256'].items()
            if not (ROOT/name).is_file() or sha(ROOT/name) != digest])
        evidence_paths.append(path)
    providers = []
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        for prefix, key in [('atom', 'atom_id'), ('artifact', 'artifact_id')]:
            rows = db.execute(f'''SELECT a.{key},a.fqdn,a.status,a.is_publishable,
                v.version_id::text,v.trust_tier,{prefix}_is_publishable(a.{key}) AS native_eligible
                FROM {prefix}s a JOIN {prefix}_versions v USING({key})
                WHERE v.is_latest AND a.fqdn LIKE %s ORDER BY a.fqdn''',
                ('sciona.atoms.ml.cassava_execution.%',)).fetchall()
            for row in rows:
                identity = row.pop(key)
                row['catalog'] = prefix
                row['metadata_counts'] = {suffix: db.execute(
                    f'SELECT count(*) AS n FROM {prefix}_{suffix} WHERE {key}=%s',
                    (identity,)).fetchone()['n'] for suffix in ('descriptions', 'parameters', 'references', 'audit_rollups')}
                providers.append(row)
    return dict(format='cassava-current-readiness.v1', catalog_mutations=0, approved=False,
        package_count=len(environment['packages']), package_mismatches=package_mismatches,
        historical_evidence=reports, model_artifact_availability_checked=False,
        provider_state=providers,
        next_requirements=['Re-execute all sixteen trainable fits and frozen CropNet inference on the current runner after confirming the complete runtime and artifact dependencies.',
            'Expose reusable operations and source recipe boundaries without reducing the complete computation.',
            'Qualify exact provider contracts and restore native catalog eligibility.',
            'Preserve original intake identity and history through corrected-version publication.'],
        limitations=['Historical passing reports do not qualify changed runtime files.',
            'Package version matching does not by itself revalidate installed file contents.',
            'Local external model artifacts remain caller-managed and are not redistributed.'],
        evidence_sha256={str(p.relative_to(ROOT)): sha(p) for p in evidence_paths})


if __name__ == '__main__':
    result = audit()
    (ROOT/'docs/reviews/decomposed_cassava_readiness.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('catalog_mutations', 'package_count', 'package_mismatches',
        'historical_evidence', 'model_artifact_availability_checked')}))
