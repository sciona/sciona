"""Attach immutable existing verification reports to four already-approved originals."""
import argparse
import json
import subprocess
from pathlib import Path
from scripts.import_flavours_provider_drafts import ROOT, connect
from scripts.import_residual_execution_drafts import ensure_row, comparable
from scripts.stage_remaining_legacy_physics_revisions import history
from scripts.audit_remaining_cdg_readiness import inspect
from scripts.promote_arc_execution import sha

COMMIT='64d5f7e2'
REPORTS=['ensemble_corrected_intake_served_verification','nasa_corrected_intake_served_verification',
         'competition_nasa_first_corrected_served','competition_otto_decomposed_served_selection']
GATES=ROOT/'docs/reviews/competition_reference_repair_gates.json'


def records():
    result=[]
    commit=subprocess.check_output(['git','rev-parse',COMMIT],cwd=ROOT,text=True).strip()
    for name in REPORTS:
        path='docs/reviews/'+name+'.json'
        raw=subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)
        if raw!=(ROOT/path).read_bytes():raise ValueError('Immutable prior verification changed')
        report=json.loads(raw)
        if report['passed'] is not True:raise ValueError('Passing prior verification required')
        row=report['execution'] if name=='nasa_corrected_intake_served_verification' else report
        if row.get('passed') is not True:raise ValueError('Passing execution report required')
        result.append(dict(artifact_id=row['artifact_id'],version_id=row.get('version_id',row.get('selected_version_id')),
            content_hash=row.get('graph_sha256',row.get('graph_digest')),ref_id='verified-original-'+row['artifact_id'],
            url='https://github.com/sciona/sciona/blob/'+commit+'/'+path,report_sha256=sha(ROOT/path)))
    return result


def snapshot(db,items):
    result={}
    for r in items:
        identity=r['artifact_id']
        result[identity]={}
        for table in ('artifacts','artifact_references'):
            result[identity][table]=comparable(db.execute(f'SELECT to_jsonb(t) AS row FROM {table} t WHERE artifact_id=%s ORDER BY to_jsonb(t)::text',(identity,)).fetchall())
        result[identity]['reference']=comparable(db.execute('SELECT to_jsonb(t) AS row FROM references_registry t WHERE ref_id=%s',(r['ref_id'],)).fetchall())
        result[identity]['history']=history(db,r['version_id'])
    return result


class InjectedFailure(RuntimeError):pass


def repair(apply=False,fail_after=None):
    items=records();writes=0
    if apply:
        gates=json.loads(GATES.read_text())
        if gates['passed'] is not True or gates['script_sha256']!=sha(Path(__file__)) or gates['records']!=items:
            raise ValueError('Current rollback qualification required')
    with connect() as db:
        db.execute("SELECT pg_advisory_xact_lock(hashtext('competition-reference-repair.v1'))")
        for r in items:
            db.execute('SELECT artifact_id FROM artifacts WHERE artifact_id=%s FOR UPDATE',(r['artifact_id'],)).fetchone()
            before=inspect(db,r['artifact_id'])
            if (before['status']!='approved' or not before['served'] or not before['is_publishable'] or before['trust_tier']!=3
                or before['version_id']!=r['version_id'] or before['content_hash']!=r['content_hash']
                or before['failed_selected_audits'] or before['missing_native_requirements'] not in ([],['reference'])):
                raise ValueError('Exact prior approved graph state required')
            preserved=history(db,r['version_id'])
            title='Stored execution and served selection verification'
            inserts=[('references_registry',dict(ref_id=r['ref_id']),dict(ref_id=r['ref_id'],ref_type='web',title=title,url=r['url'])),
                ('artifact_references',dict(artifact_id=r['artifact_id'],ref_key=r['ref_id']),dict(artifact_id=r['artifact_id'],
                    ref_key=r['ref_id'],ref_id=r['ref_id'],title=title,url=r['url'],source='llm_extracted',verified=True,confidence='high',
                    relevance_note='Existing immutable verification for this exact selected version and graph. Report SHA256 '+r['report_sha256']+'. Metadata link repair only; no new runtime or semantic qualification.'))]
            for table,keys,row in inserts:
                writes+=int(ensure_row(db,table,keys,row))
                if fail_after is not None and writes>=fail_after:raise InjectedFailure('Injected reference repair failure')
            after=inspect(db,r['artifact_id'])
            expected=dict(before,native_eligible=True,missing_native_requirements=[])
            if after!=expected or history(db,r['version_id'])!=preserved:
                raise ValueError('Eligibility repair changed unrelated selected state')
        if fail_after is not None:raise ValueError('Fault point not reached')
        if not apply:db.rollback()
    return dict(applied=apply,writes=writes,native_eligible_originals=4,existing_approvals_preserved=True,
                runtime_rerun=False,records=items)


def validate():
    items=records()
    with connect() as db:before=snapshot(db,items)
    dry=repair()
    def unchanged():
        with connect() as db:
            if snapshot(db,items)!=before:raise ValueError('Repair survived rollback')
    unchanged()
    points=[1,4,8]
    for point in points:
        try:repair(fail_after=point)
        except InjectedFailure:pass
        else:raise ValueError('Expected injected rollback')
        unchanged()
    return dict(passed=True,catalog_mutations=0,rollback_points=points,dry_run=dry,records=items,script_sha256=sha(Path(__file__)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['validate','apply','repeat'])
    action=parser.parse_args().action
    result=validate() if action=='validate' else repair(apply=True)
    if action=='repeat' and result['writes']:raise ValueError('Repeat changed references')
    path=GATES if action=='validate' else ROOT/('docs/reviews/competition_reference_repair_'+action+'.json')
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,rollback_points=result['rollback_points']) if action=='validate' else {k:v for k,v in result.items() if k!='records'}))
