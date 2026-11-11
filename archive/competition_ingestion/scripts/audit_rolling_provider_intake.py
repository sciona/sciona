"""Identify the existing rolling implementation for canonical identity mirroring."""
import inspect
import json
import subprocess
from pathlib import Path
from sciona.atoms.ml.tabular.gradient_boosting import atoms
from scripts.audit_decomposed_future_sales_readiness import ROOT,sha
from scripts.import_flavours_provider_drafts import connect

FQDN='sciona.atoms.ml.tabular.gradient_boosting.rolling_statistics'
IDENTITY='9dc67af4-d131-42dc-8bb3-dd3b2e1a7009'
VERSION='8ff9681f-f14e-57dd-9237-c34193cdc3a0'
HASH='60db40fb38e90f530117dbbb8447664c6ef77ac6d38514f345f3bc77837abeff'


def audit():
    path='docs/reviews/decomposed_future_sales_interfaces.json'
    raw=(ROOT/path).read_bytes()
    if raw!=subprocess.check_output(['git','show','86ae033e:'+path],cwd=ROOT):
        raise ValueError('Immutable interface review changed')
    review=json.loads(raw)
    if any(sha(ROOT/p)!=h for p,h in review['code_sha256'].items()):raise ValueError('Interface reviewer changed')
    executed=ROOT/'docs/reviews/decomposed_future_sales_graph_execution.json'
    if sha(executed)!=review['execution_evidence_sha256']:raise ValueError('Execution evidence changed')
    source=Path(inspect.getfile(inspect.unwrap(atoms.rolling_statistics)))
    if sha(source)!=json.loads(executed.read_text())['provider_sha256'][atoms.rolling_statistics.__module__+'.rolling_statistics']:
        raise ValueError('Executed rolling implementation changed')
    if list(inspect.signature(atoms.rolling_statistics).parameters)!=['values','window_size','agg_fns']:
        raise ValueError('Rolling interface changed')
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        rows=db.execute('''SELECT a.atom_id::text,a.fqdn,a.import_module,a.source_symbol,a.status,a.is_publishable,
            v.version_id::text,v.content_hash,v.trust_tier,atom_is_publishable(a.atom_id) AS native_eligible
            FROM atoms a JOIN atom_versions v USING(atom_id) WHERE a.fqdn=%s AND v.is_latest''',(FQDN,)).fetchall()
        if len(rows)!=1:raise ValueError('Unique selected legacy version required')
        old=rows[0]
        if old!=dict(atom_id=IDENTITY,fqdn=FQDN,import_module='sciona.atoms.ml.tabular.gradient_boosting.atoms',
            source_symbol='rolling_statistics',status='approved',is_publishable=True,version_id=VERSION,
            content_hash=HASH,trust_tier=3,native_eligible=True):raise ValueError('Existing rolling identity or approval changed')
        canonical=db.execute('SELECT artifact_id::text,fqdn,status,is_publishable FROM artifacts WHERE fqdn=%s OR artifact_id=%s',(FQDN,IDENTITY)).fetchall()
        if len(canonical)>1 or (canonical and (canonical[0]['artifact_id']!=IDENTITY or canonical[0]['fqdn']!=FQDN)):
            raise ValueError('Conflicting canonical identity')
    return dict(metadata_only_identity_eligible=True,providers=[dict(existing=old,canonical=canonical,source_sha256=sha(source))],
                audit_sha256=sha(__file__),state_source_sha256=sha(source),catalog_mutations=0)
