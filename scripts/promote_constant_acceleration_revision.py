"""Activate the qualified original reconstructed constant_acceleration revision at automated Tier 3."""
import argparse
import json
from pathlib import Path
from uuid import UUID,uuid5

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from scripts.check_constant_acceleration_revision import check
from scripts.stage_constant_acceleration_original_revision import ROOT,ARTIFACT,EXECUTION,sha,require
from scripts.review_constant_acceleration_revision import review,LIMITATIONS
from scripts.import_residual_execution_drafts import ensure_row
from sciona.cdg_projection import build_published_cdg_projection

RUNNER='constant_acceleration-original-revision-community.v1'
DESCRIPTION='Evaluate signed final velocity, displacement and average velocity under constant acceleration. Reuses one approved provider with exact rational intermediate arithmetic and independent float64 output rounding. Three SI input arrays have identical nonempty shapes and nonnegative elapsed time. Automated Tier 3; source corrections and quotient-domain extensions remain explicit.'
PLAIN_DESCRIPTION='Calculate velocity, signed displacement and average velocity from initial velocity, constant acceleration and elapsed time in SI units. Supports reversal, zero acceleration and zero duration. Displacement is not total distance traveled; acceleration must remain constant.'
REFERENCE='openstax-constant-acceleration'
DOMAIN='Signed one-dimensional constant-acceleration motion with finite real SI inputs of identical nonempty shapes and nonnegative elapsed time. Zero-time average velocity is the continuous extension of initial velocity; zero acceleration is supported independently of source division branches. Signed displacement is not path length. Exact converted-input arithmetic before independent float64 rounding; no variable acceleration or inverse-sign inference.'


def reference(db):
    rows=db.execute('SELECT r.ref_id,r.ref_key,r.title,r.url,r.source,r.verified,r.confidence,r.relevance_note FROM artifact_references r JOIN artifact_versions v ON v.artifact_id=r.artifact_id WHERE v.version_id=%s AND r.ref_key=%s AND r.verified',(EXECUTION,REFERENCE)).fetchall()
    require(len(rows)==1,'Verified qualified-parent reference required')
    return rows[0]


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
    require(bounds==[dict(review_status='automated_pass',validity_statement=DOMAIN)],'Version domain review differs')
    expected=reference(db)
    actual=db.execute('SELECT ref_id,ref_key,title,url,source,verified,confidence,relevance_note FROM artifact_references WHERE artifact_id=%s AND ref_key=%s',(ARTIFACT,REFERENCE)).fetchall()
    require(actual==[expected],'Published scientific reference differs')
    return graph,imported


def promote(source,apply=False):
    if apply:
        gates=json.loads((ROOT/'docs/reviews/constant_acceleration_revision_publication_transaction.json').read_text())
        require(gates['passed'] and gates['post_activation_failure_rolled_back']
            and gates['publisher_sha256']==sha(__file__)
            and gates['validator_sha256']==sha(ROOT/'scripts/validate_constant_acceleration_publication_transaction.py'),'Publication rollback qualification required')
    qualification=review(source);created=0;updated=0
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c statement_timeout=30000 -c lock_timeout=10000') as db:
        db.execute("SELECT pg_advisory_xact_lock(hashtext('constant_acceleration-original-revision-draft.v1'))")
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
        ensure('artifact_references',dict(artifact_id=target,ref_key=REFERENCE),dict(artifact_id=target,**reference(db)))
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
        bound=uuid5(version,RUNNER+':ideal-constant_acceleration-regime')
        ensure('artifact_validity_bounds',dict(bound_id=bound),dict(bound_id=bound,artifact_id=target,version_id=version,
            scope='version',bound_kind='regime',review_status='automated_pass',evidence_ref_key=REFERENCE,
            validity_statement=DOMAIN,
            metadata=Jsonb(dict(approval_evidence_id=str(evidence),review_source='automated',scope='Runtime-enforced numerical regime; caller establishes physical applicability; no human certification.'))))
        if not already:
            updated+=db.execute("UPDATE artifacts SET status='approved',is_publishable=true,description=%s,verified_leaf_coverage=1,leaf_count=1,top_level_input_arity=3,top_level_output_arity=3,topo_hash=%s WHERE artifact_id=%s",(DESCRIPTION,topology,target)).rowcount
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
        name='constant_acceleration_revision_publication_repeat.json' if report['already_approved'] else 'constant_acceleration_revision_publication.json'
        (ROOT/'docs/reviews'/name).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
