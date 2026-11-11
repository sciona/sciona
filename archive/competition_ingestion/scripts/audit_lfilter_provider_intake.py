"""Verify the existing filter registration before canonical identity mirroring."""
import inspect
import json
import subprocess
from pathlib import Path
from sciona.atoms.scipy import signal as atoms
from sciona.atoms.provider_inventory import ProviderRepo
from sciona.atoms.supabase_seed import _parse_registered_atoms
from scripts.audit_decomposed_vsb_readiness import ROOT,sha
from scripts.import_flavours_provider_drafts import connect

FQDN='sciona.atoms.scipy.signal.lfilter'
IDENTITY='773f66ef-aa3a-40c2-a83e-680651517011'
VERSION='b2d4c1c2-9e7b-5ce4-ac66-c8b14fa864e6'
HASH='c4b6387d6fee049494a725d9530a14ff4bac82b8f57c16c0b69516554c10bdd2'


def audit():
    path='docs/reviews/vsb_lfilter_reuse.json'
    raw=(ROOT/path).read_bytes()
    if raw!=subprocess.check_output(['git','show','1c44c993:'+path],cwd=ROOT):
        raise ValueError('Immutable reuse comparison changed')
    review=json.loads(raw)
    source=Path(inspect.getfile(inspect.unwrap(atoms.lfilter)))
    if sha(source)!=review['provider_sha256'] or sha(ROOT/'sciona/vsb_baseline.py')!=review['baseline_sha256']:
        raise ValueError('Compared implementation changed')
    if sha(ROOT/'scripts/review_vsb_lfilter_reuse.py')!=review['reviewer_sha256']:
        raise ValueError('Reuse reviewer changed')
    directory=ROOT.parent/'sciona-atoms'
    specs=[s for s in _parse_registered_atoms(repo=ProviderRepo('sciona-atoms',directory),artifact_root=directory/'src/sciona/atoms') if s.fqdn==FQDN]
    if len(specs)!=1:raise ValueError('Unique local registration required')
    spec=specs[0]
    if str(spec.version_id)!=VERSION or spec.content_hash!=HASH or spec.file_path.resolve()!=source.resolve():
        raise ValueError('Registered source fingerprint differs from existing version')
    signature=inspect.signature(atoms.lfilter)
    if list(signature.parameters)!=['b','a','x','axis','zi'] or signature.parameters['axis'].default!=-1 or signature.parameters['zi'].default is not None:
        raise ValueError('Filter interface changed')
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        rows=db.execute('''SELECT a.atom_id::text,a.fqdn,a.import_module,a.source_symbol,a.status,a.is_publishable,
            v.version_id::text,v.content_hash,v.trust_tier,atom_is_publishable(a.atom_id) AS native_eligible
            FROM atoms a JOIN atom_versions v USING(atom_id) WHERE a.fqdn=%s AND v.is_latest''',(FQDN,)).fetchall()
        expected=dict(atom_id=IDENTITY,fqdn=FQDN,import_module='sciona.atoms.scipy.signal',source_symbol='lfilter',
            status='approved',is_publishable=True,version_id=VERSION,content_hash=HASH,trust_tier=3,native_eligible=True)
        if rows!=[expected]:raise ValueError('Existing filter identity or approval changed')
        fingerprint=db.execute('SELECT fingerprint FROM atom_versions WHERE version_id=%s',(VERSION,)).fetchone()['fingerprint']
        if fingerprint!=spec.fingerprint:raise ValueError('Stored and parsed source fingerprint differ')
        canonical=db.execute('SELECT artifact_id::text,fqdn,status,is_publishable FROM artifacts WHERE fqdn=%s OR artifact_id=%s',(FQDN,IDENTITY)).fetchall()
        if len(canonical)>1 or (canonical and (canonical[0]['artifact_id']!=IDENTITY or canonical[0]['fqdn']!=FQDN)):
            raise ValueError('Conflicting canonical identity')
    return dict(metadata_only_identity_eligible=True,providers=[dict(existing=expected,canonical=canonical,source_sha256=sha(source))],
        audit_sha256=sha(__file__),state_source_sha256=sha(source),catalog_mutations=0)
