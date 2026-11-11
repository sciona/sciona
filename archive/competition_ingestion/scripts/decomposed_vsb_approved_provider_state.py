"""Freeze and verify exact approved providers independently of original CDG state."""
import json
import subprocess
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5
from sciona.decomposed_vsb_graph import build_decomposed_vsb_graph
from scripts.stage_vsb_decomposed_providers import ROOT, review, sha, require
from scripts.promote_vsb_decomposed_providers import snapshot, PORT_FIELDS
from scripts.import_flavours_provider_drafts import connect

PLAN = ROOT/'docs/reviews/vsb_approved_provider_state.json'


def records():
    review()
    path='docs/reviews/vsb_decomposed_provider_publication_plan.json'
    expected=subprocess.check_output(['git','show','fe517291:'+path],cwd=ROOT)
    require((ROOT/path).read_bytes()==expected,'Immutable provider publication changed')
    publication=json.loads(expected)
    result=[dict(r,ports=r['ports']['artifact_io_specs'],legacy_ports=r['ports']['atom_io_specs']) for r in publication['records']]
    filter_path='docs/reviews/lfilter_provider_publication_plan.json'
    raw=subprocess.check_output(['git','show','922c32ca:'+filter_path],cwd=ROOT)
    require((ROOT/filter_path).read_bytes()==raw,'Immutable filter publication changed')
    reused=json.loads(raw)['records'][0]
    result.append(dict(reused,runtime_fqdn='sciona.atoms.scipy.signal.lfilter',
        catalog_fqdn=reused['fqdn'],ports=reused['interfaces'],legacy_ports=[]))
    require(len(result)==23 and len({r['artifact_id'] for r in result})==23,'Twenty-three distinct qualified providers required')
    require({r['runtime_fqdn'] for r in result}=={n.matched_primitive for n in build_decomposed_vsb_graph().nodes},
            'Exact graph provider inventory required')
    return sorted(result,key=lambda r:r['runtime_fqdn'])


def check(db, plan):
    ids = [UUID(r['artifact_id']) for r in plan['records']]
    for r in plan['records']:
        for prefix, key in [('atom','atom_id'), ('artifact','artifact_id')]:
            rows = db.execute(f'''SELECT a.fqdn,a.status,a.is_publishable,v.version_id::text,v.content_hash,v.trust_tier
                FROM {prefix}s a JOIN {prefix}_versions v USING({key}) WHERE a.{key}=%s AND v.is_latest''', (UUID(r['artifact_id']),)).fetchall()
            require(rows == [dict(fqdn=r['catalog_fqdn'],status='approved',is_publishable=True,
                version_id=r['version_id'],content_hash=r['content_hash'],trust_tier=3)], 'Exact approved provider version differs')
            ports = db.execute('SELECT '+','.join(PORT_FIELDS)+f' FROM {prefix}_io_specs WHERE version_id=%s ORDER BY direction,ordinal', (UUID(r['version_id']),)).fetchall()
            require(ports == (r['legacy_ports'] if prefix=='atom' else r['ports']), 'Qualified provider interface differs')
            require(db.execute(f'SELECT {prefix}_is_publishable(%s) AS yes', (UUID(r['artifact_id']),)).fetchone()['yes'], 'Provider not natively eligible')
        require(not db.execute('SELECT 1 FROM artifact_dependencies WHERE dependent_version_id=%s', (UUID(r['version_id']),)).fetchone(), 'Unexpected internal provider dependency')
        require(not db.execute('SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND NOT passed', (UUID(r['version_id']),)).fetchone(), 'Failed provider audit')
    if 'catalog_snapshot' in plan:
        require(snapshot(db, ids) == plan['catalog_snapshot'], 'Approved provider metadata or history changed')
    for table, key in [('catalog_atoms_served','atom_id'),('catalog_artifacts_served','artifact_id')]:
        require(db.execute(f'SELECT count(DISTINCT {key}) AS n FROM {table} WHERE {key}=ANY(%s)', (ids,)).fetchone()['n'] == 23, 'Not all providers served')


def qualify_plan():
    frozen = json.loads(PLAN.read_text())
    require(frozen['records'] == records() and frozen['checker_sha256'] == sha(Path(__file__)), 'Approved provider qualification drift')
    return frozen


def contracts():
    return {r['runtime_fqdn']:r for r in qualify_plan()['records']}


if __name__ == '__main__':
    from scripts import promote_vsb_decomposed_providers as publication
    published=publication.qualify_plan()
    result=dict(passed=True,records=records(),checker_sha256=sha(Path(__file__)))
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        publication.check(db,published,published=True)
        check(db,result)
        result['catalog_snapshot']=snapshot(db,[UUID(r['artifact_id']) for r in result['records']])
    PLAN.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(approved_providers=23,catalog_mutations=0)))
