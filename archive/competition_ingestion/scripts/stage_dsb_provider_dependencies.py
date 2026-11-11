"""Stage exact DSB dependency declarations without approving providers."""
import argparse
import json
from pathlib import Path
from uuid import UUID
from psycopg.types.json import Jsonb

from scripts.inventory_dsb_decomposed_providers import inventory, ROOT, sha
from scripts.plan_dsb_provider_dependencies import plan as dependency_plan
from scripts.audit_dsb_static_dependencies import audit
from scripts.complete_dsb_draft_metadata import protected
from scripts.import_flavours_provider_drafts import connect
from scripts.import_residual_execution_drafts import ensure_row, comparable

PLAN=ROOT/'docs/reviews/dsb_dependency_intake_plan.json'
RUNNER='dsb-provider-dependency-intake.v1'
FIELDS=('dependent_version_id','dependency_artifact_fqdn','dependency_content_hash',
        'port_name','dependency_role','optional','binding_metadata')
KEYS=FIELDS[:4]


def expected():
    reviewed=dependency_plan();static=audit()
    path=ROOT/'docs/reviews/dsb_static_dependency_audit.json'
    if static!=json.loads(path.read_text()):raise ValueError('Static dependency evidence drift')
    path=ROOT/'docs/reviews/dsb_provider_dependency_plan.json'
    if reviewed!=json.loads(path.read_text()):raise ValueError('Dependency plan drift')
    current=inventory()
    records={r['runtime_fqdn']:r for r in current['providers']}
    identities=[]
    for r in records.values():
        ids=[]
        for prefix in ('atom','artifact'):
            latest=r[prefix+'_latest']
            if (len(latest)!=1 or latest[0]['version_id']!=r['version_id']
                    or latest[0]['content_hash']!=r['content_hash'] or latest[0]['trust_tier']!=3):
                raise ValueError('Exact provider version missing')
            ids.append(latest[0]['identity'])
        if ids[0]!=ids[1]:raise ValueError('Catalog identities disagree')
        identities.append(ids[0])
    rows=[]
    for edge in reviewed['dependencies']:
        target=records[edge['callee']]
        rows.append(dict(dependent_version_id=edge['caller_version'],
            dependency_artifact_fqdn=target['catalog_fqdn'],dependency_content_hash=target['content_hash'],
            port_name='',dependency_role='logic_atom',optional=False,
            binding_metadata=dict(runtime_fqdn=edge['callee'],provider_version_id=edge['callee_version'],
                dependency_evidence=edge,graph_digest=current['graph_digest'],
                scope='Dependency of the qualified DSB graph bindings, including returned factories and models. Arbitrary caller callbacks or replacement runtime objects are outside this evidence.',
                review_status='draft_dependency_declaration_not_publication_approval')))
    if len(rows)!=40 or len(identities)!=34:raise ValueError('Reviewed dependency scope changed')
    return rows,identities


def current_rows(db,versions):
    rows=db.execute('SELECT '+','.join(FIELDS)+' FROM artifact_dependencies WHERE dependent_version_id=ANY(%s)',
                    (list(map(UUID,versions)),)).fetchall()
    return sorted(comparable(rows),key=lambda r:tuple(r[k] for k in KEYS))


def plan():
    rows,identities=expected();versions=sorted({r['dependent_version_id'] for r in rows})
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        before=current_rows(db,versions)
        if before:raise ValueError('Existing declarations require reconciliation')
        snapshot=protected(db,list(map(UUID,identities)))
    return dict(approved=False,catalog_mutations=0,rows=rows,identities=identities,
        versions=versions,preserved=snapshot,script_sha256=sha(Path(__file__)),
        dependency_plan_sha256=sha(ROOT/'docs/reviews/dsb_provider_dependency_plan.json'),
        static_audit_sha256=sha(ROOT/'docs/reviews/dsb_static_dependency_audit.json'))


class InjectedFailure(RuntimeError):pass


def stage(apply=False,fail_after=None):
    frozen=json.loads(PLAN.read_text());rows,identities=expected()
    if rows!=frozen['rows'] or identities!=frozen['identities'] or sha(Path(__file__))!=frozen['script_sha256']:
        raise ValueError('Frozen dependency intake changed')
    for field,name in [('dependency_plan_sha256','dsb_provider_dependency_plan.json'),('static_audit_sha256','dsb_static_dependency_audit.json')]:
        if sha(ROOT/'docs/reviews'/name)!=frozen[field]:raise ValueError('Dependency evidence changed')
    writes=0
    with connect() as db:
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(RUNNER,))
        if protected(db,list(map(UUID,identities)))!=frozen['preserved']:
            raise ValueError('Protected publication or history drift')
        for row in rows:
            writes+=int(ensure_row(db,'artifact_dependencies',{k:row[k] for k in KEYS},
                dict(row,binding_metadata=Jsonb(row['binding_metadata']))))
            if writes==fail_after:raise InjectedFailure('Injected dependency failure')
        if current_rows(db,frozen['versions'])!=sorted(rows,key=lambda r:tuple(r[k] for k in KEYS)):
            raise ValueError('Stored declarations differ or contain unexpected links')
        if protected(db,list(map(UUID,identities)))!=frozen['preserved']:
            raise ValueError('Dependency insertion changed publication or history')
        if fail_after is not None:raise ValueError('Failure injection not reached')
        if not apply:db.rollback()
    return dict(applied=apply,approved=False,dependency_links=40,rows_created=writes,
                publication_and_history_preserved=True,plan_sha256=sha(PLAN))


def validate():
    frozen=json.loads(PLAN.read_text())
    def unchanged():
        with connect() as db:
            if current_rows(db,frozen['versions']):raise ValueError('Rollback left dependencies')
            if protected(db,list(map(UUID,frozen['identities'])))!=frozen['preserved']:
                raise ValueError('Rollback changed protected state')
    unchanged();dry=stage();unchanged()
    for point in (1,20,40):
        try:stage(fail_after=point)
        except InjectedFailure:pass
        else:raise ValueError('Expected failure')
        unchanged()
    return dict(passed=True,catalog_mutations=0,dry_run=dry,rollback_points=[1,20,40],
                plan_sha256=sha(PLAN),script_sha256=sha(Path(__file__)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['plan','validate','apply','repeat'])
    action=parser.parse_args().action
    if action=='plan':report=plan();path=PLAN
    elif action=='validate':report=validate();path=ROOT/'docs/reviews/dsb_dependency_intake_transactions.json'
    else:
        gate=json.loads((ROOT/'docs/reviews/dsb_dependency_intake_transactions.json').read_text())
        if not gate['passed'] or gate['plan_sha256']!=sha(PLAN) or gate['script_sha256']!=sha(Path(__file__)):
            raise ValueError('Matching transaction validation required')
        report=stage(apply=True)
        if action=='repeat' and report['rows_created']!=0:raise ValueError('Non-idempotent repeat')
        path=ROOT/f'docs/reviews/dsb_dependency_intake_{action}.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','identities','versions','preserved')}))
