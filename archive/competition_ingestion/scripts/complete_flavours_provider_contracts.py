"""Complete exact provider dependency links and missing legacy ports, without approval."""
import argparse
import hashlib
import json
from uuid import UUID

from psycopg.types.json import Jsonb

from scripts.import_flavours_provider_drafts import connect, reviewed_plan, ROOT
from scripts.import_residual_execution_drafts import ensure_row, comparable
from scripts.inventory_flavours_decomposed_providers import inventory
from scripts.plan_flavours_provider_intake import sha

PLAN = ROOT/'docs/reviews/flavours_provider_contract_plan.json'
RUNNER = 'flavours-provider-contract-completion.v1'
PORT_FIELDS = ('direction','name','ordinal','type_desc','required','default_value_repr','constraints')


def snapshot(db, identities):
    """Protect publication, version history, canonical interfaces and served rows."""
    result = {}
    for table, key in [('atoms','atom_id'),('atom_versions','atom_id'),
                       ('artifacts','artifact_id'),('artifact_versions','artifact_id'),
                       ('artifact_io_specs','artifact_id'),('catalog_artifacts_served','artifact_id')]:
        # Legacy port insertion refreshes atoms.updated_at through the database
        # trigger. Preserve every substantive field while retaining that honest
        # metadata timestamp; never reset it to conceal the port addition.
        value = "to_jsonb(t)-'updated_at'" if table=='atoms' else 'to_jsonb(t)'
        rows = db.execute(f'SELECT {value} AS row FROM {table} t WHERE {key}=ANY(%s) ORDER BY {value}::text', (identities,)).fetchall()
        result[table] = hashlib.sha256(json.dumps(rows,sort_keys=True,default=str).encode()).hexdigest()
    return result


def expected_dependencies(records):
    by_runtime = {r['runtime_fqdn']:r for r in records}
    result = []
    for record in records:
        for runtime in record['dependencies']:
            target = by_runtime[runtime]
            result.append(dict(dependent_version_id=record['version_id'],
                dependency_artifact_fqdn=target['catalog_fqdn'],dependency_content_hash=target['content_hash'],
                port_name='',dependency_role='logic_atom',optional=False,
                binding_metadata=dict(runtime_fqdn=runtime,provider_version_id=target['version_id'],
                    scope='Required registered implementation call; exact source closure qualified by full evaluated execution.')))
    return result


def plan():
    reviewed_plan()  # Recheck frozen implementation, source and execution evidence.
    current = inventory()
    if current['closure_providers'] != 39 or current['registered_dependency_links'] != 11:
        raise ValueError('Reviewed closure differs')
    mirrors = json.loads((ROOT/'docs/reviews/flavours_legacy_reconciliation_plan.json').read_text())
    additions = []
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        identities = []
        for record in current['providers']:
            for catalog in ('legacy','canonical'):
                versions = record[catalog+'_latest']
                if len(versions)!=1 or versions[0]['version_id']!=record['version_id'] or versions[0]['content_hash']!=record['content_hash'] or versions[0]['trust_tier']!=3:
                    raise ValueError('Exact staged provider missing')
            identity = db.execute('SELECT artifact_id FROM artifact_versions WHERE version_id=%s',(record['version_id'],)).fetchone()['artifact_id']
            identities.append(identity)
        for record in mirrors['providers']:
            identity, version = UUID(record['artifact_id']), UUID(record['version_id'])
            canonical = db.execute('SELECT '+','.join(PORT_FIELDS)+' FROM artifact_io_specs WHERE artifact_id=%s AND version_id=%s ORDER BY direction,ordinal',(identity,version)).fetchall()
            expected = sorted(record['ports'],key=lambda p:(p['direction'],p['ordinal']))
            if comparable(canonical)!=expected:
                raise ValueError('Canonical native ports differ')
            if db.execute('SELECT 1 FROM atom_io_specs WHERE atom_id=%s AND version_id=%s',(identity,version)).fetchone():
                raise ValueError('Legacy interface is no longer absent')
            additions.extend(dict(atom_id=str(identity),version_id=str(version),**port) for port in expected)
        dependencies = expected_dependencies(current['providers'])
        for row in dependencies:
            if db.execute('SELECT 1 FROM artifact_dependencies WHERE dependent_version_id=%s AND dependency_artifact_fqdn=%s',
                          (row['dependent_version_id'],row['dependency_artifact_fqdn'])).fetchone():
                raise ValueError('Dependency already recorded; review existing declaration')
        before = snapshot(db,identities)
    return dict(passed=True,approved=False,identities=list(map(str,identities)),
        graph_digest=current['graph_digest'],providers=current['providers'],legacy_ports=additions,
        dependencies=dependencies,preserved_snapshot=before,
        execution_evidence_sha256=sha(ROOT/'docs/reviews/flavours_decomposed_evaluation_execution.json'))


class InjectedFailure(RuntimeError):
    pass


def stage(apply=False, fail_after=None):
    frozen = json.loads(PLAN.read_text())
    reviewed_plan()
    current = inventory()
    if current['providers']!=frozen['providers'] or current['graph_digest']!=frozen['graph_digest']:
        raise ValueError('Provider closure or publication drift')
    if sha(ROOT/'docs/reviews/flavours_decomposed_evaluation_execution.json')!=frozen['execution_evidence_sha256']:
        raise ValueError('Execution evidence drift')
    if expected_dependencies(current['providers'])!=frozen['dependencies']:
        raise ValueError('Dependency plan differs from current registered call closure')
    writes = 0
    identities = list(map(UUID,frozen['identities']))
    with connect() as db:
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(RUNNER,))
        if snapshot(db,identities)!=frozen['preserved_snapshot']:
            raise ValueError('Protected history changed')
        legacy_states=db.execute('SELECT atom_id,is_publishable FROM atoms WHERE atom_id=ANY(%s) FOR UPDATE',(identities,)).fetchall()
        def ensure(table,key,row):
            nonlocal writes
            writes += int(ensure_row(db,table,key,row))
            if writes==fail_after:
                raise InjectedFailure('Injected contract completion failure')
        for row in frozen['legacy_ports']:
            ensure('atom_io_specs',{k:row[k] for k in ('atom_id','version_id','direction','name')},row)
        # The IO trigger computes structural publishability, which must not
        # bypass the pending semantic approval of these exact providers.
        for state in legacy_states:
            db.execute('UPDATE atoms SET is_publishable=%s WHERE atom_id=%s AND is_publishable IS DISTINCT FROM %s',
                       (state['is_publishable'],state['atom_id'],state['is_publishable']))
        for row in frozen['dependencies']:
            key = {k:row[k] for k in ('dependent_version_id','dependency_artifact_fqdn','dependency_content_hash','port_name')}
            ensure('artifact_dependencies',key,dict(row,binding_metadata=Jsonb(row['binding_metadata'])))
        if snapshot(db,identities)!=frozen['preserved_snapshot']:
            raise ValueError('Publication, canonical contracts or history changed')
        if fail_after is not None:
            raise ValueError('Failure injection point not reached')
        if not apply:
            db.rollback()
    return dict(applied=apply,approved=False,rows_created=writes,legacy_ports=len(frozen['legacy_ports']),
        dependency_links=len(frozen['dependencies']),publication_and_version_history_preserved=True,
        legacy_metadata_timestamps_may_advance=True,plan_sha256=sha(PLAN))


def validate():
    frozen=json.loads(PLAN.read_text())
    def unchanged():
        with connect() as db:
            for row in frozen['legacy_ports']:
                if db.execute('SELECT 1 FROM atom_io_specs WHERE atom_id=%s AND version_id=%s',(row['atom_id'],row['version_id'])).fetchone():
                    raise ValueError('Rollback left legacy ports')
            for row in frozen['dependencies']:
                if db.execute('SELECT 1 FROM artifact_dependencies WHERE dependent_version_id=%s AND dependency_artifact_fqdn=%s',
                              (row['dependent_version_id'],row['dependency_artifact_fqdn'])).fetchone():
                    raise ValueError('Rollback left dependencies')
            if snapshot(db,list(map(UUID,frozen['identities'])))!=frozen['preserved_snapshot']:
                raise ValueError('Rollback changed protected history')
    unchanged()
    dry=stage()
    unchanged()
    points=[1,len(frozen['legacy_ports'])+1,dry['rows_created']]
    for point in points:
        try:
            stage(fail_after=point)
        except InjectedFailure:
            pass
        else:
            raise ValueError('Expected injected failure')
        unchanged()
    return dict(passed=True,catalog_mutations=0,dry_run=dry,fault_after_writes=points,
        all_rollbacks_complete=True,plan_sha256=sha(PLAN),importer_sha256=sha(__file__))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['plan','validate','apply','repeat'])
    action=parser.parse_args().action
    if action=='plan':
        result=plan()
        path=PLAN
    elif action=='validate':
        result=validate()
        path=ROOT/'docs/reviews/flavours_provider_contract_transactions.json'
    else:
        gates=json.loads((ROOT/'docs/reviews/flavours_provider_contract_transactions.json').read_text())
        if not gates['passed'] or gates['plan_sha256']!=sha(PLAN) or gates['importer_sha256']!=sha(__file__):
            raise ValueError('Matching transaction qualification required')
        result=stage(apply=True)
        if action=='repeat' and result['rows_created']!=0:
            raise ValueError('Repeat was not idempotent')
        path=ROOT/('docs/reviews/flavours_provider_contract_'+action+'.json')
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('providers','identities','legacy_ports','dependencies','preserved_snapshot')}))
