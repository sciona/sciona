"""Read-only identity audit for the two reusable positive-log blend providers."""
import hashlib
import inspect
import json
from pathlib import Path

from sciona.atoms.ml import positive_log_blending as providers
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from scripts.import_flavours_provider_drafts import connect

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('fit_positive_log_blend', 'apply_positive_log_blend')


def audit():
    repo = ROOT.parent/'sciona-atoms-ml'
    specs = {s.source_symbol: s for s in _parse_registered_atoms(
        repo=ProviderRepo(repo.name, repo), artifact_root=repo/'src/sciona/atoms')
        if s.import_module == providers.__name__ and s.source_symbol in NAMES}
    if set(specs) != set(NAMES):
        raise ValueError('Both exact registered providers required')
    rows = []
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        for name in NAMES:
            spec = specs[name]
            catalogs = {}
            for prefix, key in [('atom', 'atom_id'), ('artifact', 'artifact_id')]:
                condition = 'a.fqdn=%s OR (a.import_module=%s AND a.source_symbol=%s)' if prefix == 'atom' else 'a.fqdn=%s OR a.artifact_id::text=ANY(%s)'
                parameters = (spec.fqdn, spec.import_module, name) if prefix == 'atom' else (spec.fqdn, [r['identity'] for r in catalogs['atom']])
                catalogs[prefix] = db.execute(f'''SELECT a.{key}::text AS identity,a.fqdn,a.status,
                    a.is_publishable,v.version_id::text,v.content_hash,v.trust_tier,v.is_latest
                    FROM {prefix}s a LEFT JOIN {prefix}_versions v USING({key})
                    WHERE {condition}
                    ORDER BY a.{key},v.version_id''', parameters).fetchall()
            rows.append(dict(runtime_fqdn=providers.__name__+'.'+name, catalog_fqdn=spec.fqdn,
                parsed_version_id=str(spec.version_id), parsed_content_hash=spec.content_hash,
                signature=str(inspect.signature(getattr(providers, name))), catalogs=catalogs))
    source = Path(providers.__file__)
    execution = ROOT/'docs/reviews/competition_mercari_tensorflow_synthetic_graph.json'
    evidence = json.loads(execution.read_text())
    if (not evidence['passed'] or not evidence['fresh_process_replay']['exact_query_predictions']
            or not all(r['registered_provider_json_parity'] for r in evidence['blend_rounds'])
            or evidence['registered_blend_provider']['sha256'] != hashlib.sha256(source.read_bytes()).hexdigest()):
        raise ValueError('Current connected provider evidence required')
    return dict(format='positive-log-blend-intake-audit.v1', catalog_mutations=0, approved=False,
        providers=rows, connected_execution_sha256=hashlib.sha256(execution.read_bytes()).hexdigest(),
        provider_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        provider_revision=evidence['registered_blend_provider']['revision'],
        limitations=['Read-only identity audit; catalog intake, full interface metadata and transactional publication gates remain outstanding.',
                    'Generic blend qualification does not approve the original competition CDG.'],
        audit_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__ == '__main__':
    result = audit()
    (ROOT/'docs/reviews/positive_log_blend_intake_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(providers=len(result['providers']), catalog_rows=sum(len(c) for p in result['providers']
        for c in p['catalogs'].values()), catalog_mutations=0)))
