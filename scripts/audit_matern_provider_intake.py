"""Verify the existing Matérn registration before canonical identity mirroring."""
import inspect
import json
import subprocess
from pathlib import Path
from sciona.atoms.ml.sklearn.gaussian_process.kernels import atoms
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from scripts.audit_decomposed_vsb_readiness import ROOT,sha
from scripts.import_flavours_provider_drafts import connect

FQDN='sciona.atoms.ml.sklearn.gaussian_process.kernels.matern_kernel_matrix'
IDENTITY='228a0adf-f685-4739-bdf4-71e0ae68835e'
VERSION='84104a39-6352-516b-b1a2-5a33a3834bc1'
HASH='17bd8119554525ece1323da65f105ff3015ac2f94d53fcdf29a16c4263d7d353'


def audit():
    path='docs/reviews/plasticc_matern_reuse.json'
    raw=(ROOT/path).read_bytes()
    if raw!=subprocess.check_output(['git','show','479b8c65:'+path],cwd=ROOT):
        raise ValueError('Immutable reuse comparison changed')
    review=json.loads(raw)
    source=Path(inspect.getfile(inspect.unwrap(atoms.matern_kernel_matrix)))
    if sha(source)!=review['provider_sha256'] or sha(ROOT/'sciona/plasticc_observations.py')!=review['observations_implementation_sha256']:
        raise ValueError('Compared implementation changed')
    if sha(ROOT/'scripts/review_plasticc_matern_reuse.py')!=review['reviewer_sha256']:
        raise ValueError('Reuse reviewer changed')
    directory=ROOT.parent/'sciona-atoms-ml'
    specs=[s for s in _parse_registered_atoms(repo=ProviderRepo('sciona-atoms-ml',directory),artifact_root=directory/'src/sciona/atoms') if s.fqdn==FQDN]
    if len(specs)!=1:raise ValueError('Unique local registration required')
    spec=specs[0]
    if str(spec.version_id)!=VERSION or spec.content_hash!=HASH or spec.file_path.resolve()!=source.resolve():
        raise ValueError('Registered source fingerprint differs from existing version')
    signature=inspect.signature(atoms.matern_kernel_matrix)
    if (list(signature.parameters)!=['X','Y','length_scale','nu'] or signature.parameters['Y'].default is not None
            or signature.parameters['length_scale'].default!=1.0 or signature.parameters['nu'].default!=1.5
            or any(signature.parameters[k].kind!=inspect.Parameter.KEYWORD_ONLY for k in ('length_scale','nu'))):
        raise ValueError('Matérn interface changed')
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        rows=db.execute('''SELECT a.atom_id::text,a.fqdn,a.import_module,a.source_symbol,a.status,a.is_publishable,
            v.version_id::text,v.content_hash,v.trust_tier,atom_is_publishable(a.atom_id) AS native_eligible
            FROM atoms a JOIN atom_versions v USING(atom_id) WHERE a.fqdn=%s AND v.is_latest''',(FQDN,)).fetchall()
        expected=dict(atom_id=IDENTITY,fqdn=FQDN,import_module='sciona.atoms.ml.sklearn.gaussian_process.kernels.atoms',source_symbol='matern_kernel_matrix',
            status='approved',is_publishable=True,version_id=VERSION,content_hash=HASH,trust_tier=3,native_eligible=True)
        if rows!=[expected]:raise ValueError('Existing kernel identity or approval changed')
        fingerprint=db.execute('SELECT fingerprint FROM atom_versions WHERE version_id=%s',(VERSION,)).fetchone()['fingerprint']
        if fingerprint!=spec.fingerprint:raise ValueError('Stored and parsed source fingerprint differ')
        canonical=db.execute('SELECT artifact_id::text,fqdn,status,is_publishable FROM artifacts WHERE fqdn=%s OR artifact_id=%s',(FQDN,IDENTITY)).fetchall()
        if len(canonical)>1 or (canonical and (canonical[0]['artifact_id']!=IDENTITY or canonical[0]['fqdn']!=FQDN)):
            raise ValueError('Conflicting canonical identity')
    return dict(metadata_only_identity_eligible=True,providers=[dict(existing=expected,canonical=canonical,source_sha256=sha(source))],
        audit_sha256=sha(__file__),state_source_sha256=sha(source),catalog_mutations=0)
