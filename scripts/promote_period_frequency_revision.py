"""Activate the qualified original period/frequency revision at automated Tier 3."""
import argparse
import json
from pathlib import Path
from uuid import UUID,uuid5

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from scripts.check_period_frequency_revision import check
from scripts.stage_period_frequency_original_revision import ROOT,ARTIFACT,sha,require
from scripts.review_period_frequency_revision import review,LIMITATIONS
from scripts.import_residual_execution_drafts import ensure_row
from sciona.cdg_projection import build_published_cdg_projection

RUNNER='period-frequency-original-revision-community.v1'
DESCRIPTION='Evaluate ordinary frequency in hertz from a supplied finite positive period in seconds using one approved reusable provider. Preserves scalar/array shape and rejects invalid inputs and floating-point overflow/underflow. Original reciprocal derivation and source history are preserved. Automated Tier 3 numerical realization.'
PLAIN_DESCRIPTION='Convert the duration of a repeating cycle into cycles per second. Reuse across periodic motion, oscillating circuits and recurring signals with explicit units. Supply the cycle duration; this operation does not estimate it from a signal or calculate angular frequency.'


def check_publication(db,qualification):
    graph,imported=check(db,True)
    row=db.execute('SELECT passed,details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version=%s AND audit_type=%s',
        (imported['version_id'],RUNNER,'semantic_audit')).fetchall()
    require(len(row)==1 and row[0]['passed'] and row[0]['details']==dict(qualification=qualification,
        publisher_sha256=sha(__file__),publication_tier=3),'Current version-bound semantic approval required')
    require(db.execute('SELECT description FROM artifacts WHERE artifact_id=%s',(ARTIFACT,)).fetchone()['description']==DESCRIPTION,'Published description differs')
    descriptions=db.execute('SELECT kind,content FROM artifact_descriptions WHERE artifact_id=%s AND language=%s ORDER BY kind',(ARTIFACT,'en')).fetchall()
    require(descriptions==[dict(kind='dejargonized',content=PLAIN_DESCRIPTION),dict(kind='technical',content=DESCRIPTION)],'Published descriptions differ')
    bounds=db.execute('SELECT review_status,validity_statement FROM artifact_validity_bounds WHERE version_id=%s',(imported['version_id'],)).fetchall()
    require(bounds==[dict(review_status='automated_pass',validity_statement='Finite positive period in seconds and finite positive ordinary frequency in hertz; both nonzero algebraic proof conditions hold.')],'Version domain review differs')
    return graph,imported


def promote(source,apply=False):
    if apply:
        gates=json.loads((ROOT/'docs/reviews/period_frequency_revision_publication_transaction.json').read_text())
        require(gates['passed'] and gates['post_activation_failure_rolled_back']
            and gates['publisher_sha256']==sha(__file__)
            and gates['validator_sha256']==sha(ROOT/'scripts/validate_period_frequency_publication_transaction.py'),'Publication rollback qualification required')
    qualification=review(source);created=0;updated=0
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c statement_timeout=30000 -c lock_timeout=10000') as db:
        db.execute("SELECT pg_advisory_xact_lock(hashtext('period-frequency-original-revision-draft.v1'))")
        state=db.execute('SELECT status,is_publishable,fqdn FROM artifacts WHERE artifact_id=%s FOR UPDATE',(ARTIFACT,)).fetchone()
        already=state['status']=='approved' and state['is_publishable']
        graph,imported=check(db,already)
        if already:check_publication(db,qualification)
        target=UUID(ARTIFACT);version=UUID(imported['version_id'])
        topology=build_published_cdg_projection(artifact=dict(artifact_id=ARTIFACT,fqdn=state['fqdn']),
            version=dict(version_id=imported['version_id'],content_hash=imported['graph_sha256']),cdg=graph).topo_hash
        def ensure(table,keys,row):
            nonlocal created
            created+=int(ensure_row(db,table,keys,row))
        require(db.execute("SELECT 1 FROM artifact_references WHERE artifact_id=%s AND ref_key='openstax-university-physics-v1-15-1' AND verified",(target,)).fetchone(),'Verified scientific reference required')
        for kind,content in [('technical',DESCRIPTION),('dejargonized',PLAIN_DESCRIPTION)]:
            ensure('artifact_descriptions',dict(artifact_id=target,kind=kind,language='en'),dict(description_id=uuid5(version,RUNNER+':'+kind),
                artifact_id=target,kind=kind,content=content,language='en',generated_by=RUNNER,reviewed=False))
        rollup=dict(overall_verdict='acceptable_with_limits',structural_status='pass',runtime_status='pass',semantic_status='pass',
            developer_semantics_status='pass',review_status='approved',review_semantic_verdict='pass',review_developer_semantics_verdict='pass',
            trust_readiness='ready',review_limitations=LIMITATIONS,review_required_actions=[],trust_blockers=[],
            acceptability_band='acceptable_with_limits',parity_coverage_level='positive_and_negative',parity_test_status='pass')
        ensure('artifact_audit_rollups',dict(artifact_id=target),dict(artifact_id=target,**rollup))
        evidence=uuid5(version,RUNNER)
        ensure('artifact_audit_evidence',dict(evidence_id=evidence),dict(evidence_id=evidence,artifact_id=target,version_id=version,
            audit_type='semantic_audit',passed=True,status='completed',source_kind='automated',runner_version=RUNNER,
            details=Jsonb(dict(qualification=qualification,publisher_sha256=sha(__file__),publication_tier=3))))
        bound=uuid5(version,RUNNER+':positive-domain')
        ensure('artifact_validity_bounds',dict(bound_id=bound),dict(bound_id=bound,artifact_id=target,version_id=version,
            scope='version',bound_kind='regime',review_status='automated_pass',evidence_ref_key='openstax-university-physics-v1-15-1',
            validity_statement='Finite positive period in seconds and finite positive ordinary frequency in hertz; both nonzero algebraic proof conditions hold.',
            metadata=Jsonb(dict(approval_evidence_id=str(evidence),review_source='automated',scope='Runtime-enforced numerical regime; no human certification.'))))
        if not already:
            updated+=db.execute("UPDATE artifacts SET status='approved',is_publishable=true,description=%s,verified_leaf_coverage=1,leaf_count=1,top_level_input_arity=1,top_level_output_arity=1,topo_hash=%s WHERE artifact_id=%s",(DESCRIPTION,topology,target)).rowcount
            updated+=db.execute('UPDATE artifact_versions SET is_latest=false WHERE artifact_id=%s AND is_latest',(target,)).rowcount
            updated+=db.execute('UPDATE artifact_versions SET is_latest=true WHERE version_id=%s AND NOT is_latest',(version,)).rowcount
        check_publication(db,qualification)
        require(review(source)==qualification,'Qualification changed during publication')
        if not apply:db.rollback()
    return dict(applied=apply,already_approved=already,rows_created=created,rows_updated=updated,trust_tier=3,
        artifact_id=ARTIFACT,version_id=imported['version_id'],graph_sha256=imported['graph_sha256'],original_history_preserved=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True);parser.add_argument('--apply',action='store_true')
    args=parser.parse_args();report=promote(args.source_directory,args.apply)
    if args.apply:
        name='period_frequency_revision_publication_repeat.json' if report['already_approved'] else 'period_frequency_revision_publication.json'
        (ROOT/'docs/reviews'/name).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
