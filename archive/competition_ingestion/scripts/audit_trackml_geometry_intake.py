"""Check existing geometry registrations before canonical identity mirroring."""
import inspect
import json
import subprocess
from pathlib import Path
from sciona.atoms.particle_tracking.helix_geometry import atoms
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from scripts.audit_decomposed_trackml_readiness import ROOT,sha
from scripts.import_flavours_provider_drafts import connect

IDENTITIES={'circle_from_three_points':'6be92ead-9c17-4678-9d94-1a5561c6f423',
            'helix_pitch_from_two_points':'4a12f880-b347-4f10-a5ef-df632453cb21',
            'helix_pitch_least_squares':'f4f2d8ae-6ea0-44a9-9768-37ebe7d63610'}


def audit():
    path='docs/reviews/trackml_geometry_reuse.json'
    raw=(ROOT/path).read_bytes()
    if raw!=subprocess.check_output(['git','show','91e0ad18:'+path],cwd=ROOT):
        raise ValueError('Immutable reuse comparison changed')
    review=json.loads(raw)
    source=Path(inspect.getfile(inspect.unwrap(atoms.circle_from_three_points)))
    if sha(source)!=review['provider_sha256']:raise ValueError('Compared provider changed')
    for name,digest in review['implementation_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Compared implementation changed')
    directory=ROOT.parent/'sciona-atoms-physics'
    specs={s.fqdn:s for s in _parse_registered_atoms(repo=ProviderRepo('sciona-atoms-physics',directory),artifact_root=directory/'src/sciona/atoms')}
    providers=[]
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        for prior in review['catalog_candidates']:
            spec=specs[prior['fqdn']];symbol=prior['source_symbol'];identity=IDENTITIES[symbol]
            if str(spec.version_id)!=prior['version_id'] or spec.content_hash!=prior['content_hash'] or spec.file_path.resolve()!=source.resolve():
                raise ValueError('Parsed provider fingerprint differs')
            rows=db.execute('''SELECT a.atom_id::text,a.fqdn,a.import_module,a.source_symbol,a.status,a.is_publishable,
                v.version_id::text,v.content_hash,v.trust_tier,atom_is_publishable(a.atom_id) AS native_eligible
                FROM atoms a JOIN atom_versions v USING(atom_id) WHERE a.fqdn=%s AND v.is_latest''',(prior['fqdn'],)).fetchall()
            expected={k:v for k,v in prior.items() if k!='served'}
            expected.update(atom_id=identity,native_eligible=True)
            if rows!=[expected]:raise ValueError('Legacy identity or approval changed')
            if db.execute('SELECT fingerprint FROM atom_versions WHERE version_id=%s',(prior['version_id'],)).fetchone()['fingerprint']!=spec.fingerprint:
                raise ValueError('Stored fingerprint differs')
            canonical=db.execute('SELECT artifact_id::text,fqdn,status,is_publishable FROM artifacts WHERE fqdn=%s OR artifact_id=%s',(prior['fqdn'],identity)).fetchall()
            if len(canonical)>1 or (canonical and (canonical[0]['artifact_id']!=identity or canonical[0]['fqdn']!=prior['fqdn'])):
                raise ValueError('Conflicting canonical identity')
            providers.append(dict(existing=expected,canonical=canonical,source_sha256=sha(source)))
    if len(providers)!=3:raise ValueError('Three exact providers required')
    return dict(metadata_only_identity_eligible=True,providers=providers,audit_sha256=sha(__file__),
                state_source_sha256=sha(source),catalog_mutations=0)
