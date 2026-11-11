"""Read-only CHAMPS prerequisites; preserve historical qualification reports."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path

from scripts.import_flavours_provider_drafts import ROOT, connect


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def audit(source):
    reviews = ROOT / 'docs/reviews'
    requirement = ROOT / 'requirements/champs-execution.txt'
    expected = dict(line.split('==') for line in requirement.read_text().splitlines()
                    if line and not line.startswith('#'))
    mismatches = []
    for name, version in expected.items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual = None
        if actual != version:
            mismatches.append(dict(package=name, expected=version, actual=actual))
    graph_path = reviews / 'competition_champs_graph_execution.json'
    graph = json.loads(graph_path.read_text())
    changed = [name for name, digest in graph['code_sha256'].items()
               if not (ROOT / name).is_file() or sha(ROOT / name) != digest]
    provider = ROOT.parent / 'sciona-atoms-ml/src/sciona/atoms/ml/champs_execution.py'
    pins_path = reviews / 'competition_champs_source_pins.json'
    pins = json.loads(pins_path.read_text())
    missing, different = [], []
    for entry in pins['pins']:
        relative = Path(entry['software_path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe public software manifest path')
        path = source / relative
        if not path.is_file():
            missing.append(relative.as_posix())
        else:
            raw = path.read_bytes()
            blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
            if hashlib.sha256(raw).hexdigest() != entry['sha256'] or blob != entry['git_blob_sha1']:
                different.append(relative.as_posix())
    providers = []
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        original = db.execute('''SELECT a.status,a.is_publishable,v.version_id::text,v.trust_tier,
            EXISTS (SELECT 1 FROM catalog_artifacts_served s WHERE s.artifact_id=a.artifact_id) AS served
            FROM artifacts a JOIN artifact_versions v USING(artifact_id)
            WHERE a.fqdn=%s AND v.is_latest''',
            ('cdg.competition.solution.kaggle.champs_molecular_properties_1st',)).fetchall()
        for prefix, key in [('atom', 'atom_id'), ('artifact', 'artifact_id')]:
            rows = db.execute(f'''SELECT a.{key},a.fqdn,a.status,a.is_publishable,
                v.version_id::text,v.trust_tier,{prefix}_is_publishable(a.{key}) AS native_eligible
                FROM {prefix}s a JOIN {prefix}_versions v USING({key})
                WHERE v.is_latest AND a.fqdn LIKE %s ORDER BY a.fqdn''',
                ('sciona.atoms.ml.champs_execution.%',)).fetchall()
            for row in rows:
                identity = row.pop(key)
                row['catalog'] = prefix
                row['metadata_counts'] = {suffix: db.execute(
                    f'SELECT count(*) AS n FROM {prefix}_{suffix} WHERE {key}=%s',
                    (identity,)).fetchone()['n'] for suffix in
                    ('descriptions', 'parameters', 'references', 'audit_rollups')}
                providers.append(row)
    return dict(format='champs-current-readiness.v1', approved=False, catalog_mutations=0,
        original_latest=original, package_mismatches=mismatches,
        source_software=dict(commit=pins['commit'], expected_files=len(pins['pins']),
                             missing=missing, mismatched=different),
        historical_graph=dict(status=graph['status'], checks=graph['checks'],
                              changed_implementation_files=changed,
                              provider_hash_matches=sha(provider) == graph['provider_sha256']),
        provider_state=providers,
        next_requirements=[
            'Qualify current dependencies and the complete pinned source software.',
            'Preserve all thirteen source model configurations, chemistry, fitted state and training lifecycle.',
            'Expose domain-independent numerical operations without embedding molecular selection policy.',
            'Validate full execution with synthetic inputs under the shared four-core budget.',
            'Restore exact provider eligibility and publish a corrected version under the original intake identity.'],
        limitations=[
            'Historical graph evidence covers one synthetic epoch per full configured model, not original training-duration or accuracy parity.',
            'Hash and package metadata checks do not replace current runtime qualification.',
            'Only pinned public software and catalog metadata are inspected; no runtime molecular records are included.'],
        evidence_sha256={str(p.relative_to(ROOT)): sha(p) for p in
                        (Path(__file__).resolve(), requirement, graph_path, pins_path)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    report = audit(parser.parse_args().source)
    (ROOT / 'docs/reviews/decomposed_champs_readiness.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in
                     ('original_latest', 'package_mismatches', 'source_software', 'historical_graph')}, indent=2))
