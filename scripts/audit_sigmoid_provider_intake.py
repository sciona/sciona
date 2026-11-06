"""Read-only identity and interface audit before reusing existing sigmoid atoms."""
import inspect
import json
from pathlib import Path

from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from sciona.atoms.ml.sklearn.calibration import atoms, state_models
from scripts.import_flavours_provider_drafts import connect
from scripts.validate_decomposed_march_baseline import ROOT, sha, runtime

NAMES = ('sigmoid_calibration_fit', 'sigmoid_calibration_predict')
PORTS = 'direction,name,ordinal,type_desc,required,default_value_repr,constraints'


def audit():
    directory = ROOT.parent/'sciona-atoms-ml'
    specs = {s.source_symbol: s for s in _parse_registered_atoms(
        repo=ProviderRepo(directory.name, directory), artifact_root=directory/'src/sciona/atoms')
        if s.import_module == atoms.__name__ and s.source_symbol in NAMES}
    if set(specs) != set(NAMES):
        raise ValueError('Both existing registered implementations required')
    providers = []
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        for name in NAMES:
            spec = specs[name]
            rows = db.execute('''SELECT a.atom_id::text,a.fqdn,a.import_module,a.source_symbol,
                a.status,a.is_publishable,v.version_id::text,v.content_hash,v.trust_tier,
                v.fingerprint FROM atoms a JOIN atom_versions v USING(atom_id)
                WHERE a.fqdn=%s AND v.is_latest''', (spec.fqdn,)).fetchall()
            if len(rows) != 1:
                raise ValueError('Unique existing latest atom required')
            row = rows[0]
            exact = (row['version_id'] == str(spec.version_id) and row['content_hash'] == spec.content_hash
                     and row['import_module'] == spec.import_module and row['source_symbol'] == name)
            ports = db.execute(f'SELECT {PORTS} FROM atom_io_specs WHERE version_id=%s ORDER BY direction,ordinal',
                               (row['version_id'],)).fetchall()
            canonical = db.execute('''SELECT artifact_id::text,fqdn,status,is_publishable FROM artifacts
                WHERE fqdn=%s OR artifact_id=%s''', (spec.fqdn, row['atom_id'])).fetchall()
            counts = {}
            for suffix in ('descriptions', 'parameters', 'references', 'audit_rollups'):
                counts[suffix] = db.execute(f'SELECT count(*) AS n FROM atom_{suffix} WHERE atom_id=%s',
                                           (row['atom_id'],)).fetchone()['n']
            parameter_versions = db.execute('''SELECT version_id::text,count(*) AS parameters
                FROM atom_parameters WHERE atom_id=%s GROUP BY version_id ORDER BY version_id''',
                (row['atom_id'],)).fetchall()
            duplicate_atoms = db.execute('''SELECT DISTINCT a.atom_id::text,a.fqdn FROM atoms a
                JOIN atom_versions v USING(atom_id) WHERE a.atom_id<>%s
                AND (v.content_hash=%s OR (a.import_module=%s AND a.source_symbol=%s)) ORDER BY a.fqdn''',
                (row['atom_id'], spec.content_hash, spec.import_module, name)).fetchall()
            fn = getattr(atoms, name)
            providers.append(dict(existing=row, parsed_version_id=str(spec.version_id),
                parsed_content_hash=spec.content_hash, exact_registered_identity=exact,
                source_sha256=sha(Path(inspect.getfile(inspect.unwrap(fn)))),
                signature=str(inspect.signature(fn)), ports=ports, metadata_counts=counts,
                parameter_version_counts=parameter_versions,
                canonical=canonical, duplicate_implementations=duplicate_atoms))
    return dict(format='sigmoid-provider-intake-audit.v1', catalog_mutations=0, approved=False,
        providers=providers, runtime=runtime(), state_source_sha256=sha(Path(inspect.getfile(state_models))),
        audit_sha256=sha(Path(__file__)),
        metadata_only_identity_eligible=all(p['exact_registered_identity'] and not p['duplicate_implementations'] for p in providers),
        limitations=['Identity and stored interface inspection only; no new publication approval.',
                    'Full parameter semantics, synthetic edge cases and transactional publication gates remain required.'])


if __name__ == '__main__':
    result = audit()
    (ROOT/'docs/reviews/sigmoid_provider_intake_audit.json').write_text(json.dumps(result, indent=2, default=str)+'\n')
    print(json.dumps(dict(providers=len(result['providers']), catalog_mutations=0,
                         metadata_only_identity_eligible=result['metadata_only_identity_eligible'])))
