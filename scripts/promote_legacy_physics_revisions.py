"""Atomically activate five qualified legacy physics revisions at Tier 3."""
import argparse
import importlib
import json
from pathlib import Path
from uuid import UUID,uuid5
import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from scripts.stage_legacy_physics_revisions import ROOT,sha,require,RUNNER as DRAFT_RUNNER
from scripts.review_legacy_physics_revisions import review
from scripts.check_legacy_physics_revision import check
from scripts.import_residual_execution_drafts import ensure_row
from sciona.cdg_projection import build_published_cdg_projection

RUNNER='legacy-physics-community.v1'


def publication_fields(db,item):
    parent=item['approved_family_qualification'];family=importlib.import_module('scripts.promote_'+item['family']+'_revision')
    bounds=db.execute('SELECT validity_statement,evidence_ref_key FROM artifact_validity_bounds WHERE version_id=%s AND review_status=%s',(parent['version_id'],'automated_pass')).fetchall()
    require(len(bounds)==1,'Unique qualified parent regime required')
    bound=bounds[0]
    refs=db.execute('SELECT ref_id,ref_key,title,url,source,verified,confidence,relevance_note FROM artifact_references WHERE artifact_id=%s AND ref_key=%s AND verified',(parent['artifact_id'],bound['evidence_ref_key'])).fetchall()
    require(len(refs)==1,'Verified qualified parent reference required')
    return family.DESCRIPTION,family.PLAIN_DESCRIPTION,bound,refs[0]


def check_publication(db,qualification):
    selected=[]
    for item in qualification['graphs']:
        graph,record=check(db,item['family'],True)
        technical,plain,bound,reference=publication_fields(db,item)
        rows=db.execute('SELECT passed,details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version=%s AND audit_type=%s',(item['version_id'],RUNNER,'semantic_audit')).fetchall()
        require(rows==[dict(passed=True,details=dict(qualification=qualification,publisher_sha256=sha(__file__),publication_tier=3))],'Legacy version-bound approval differs')
        artifact=item['artifact_id']
        require(db.execute('SELECT description FROM artifacts WHERE artifact_id=%s',(artifact,)).fetchone()['description']==technical,'Legacy published description differs')
        descriptions=db.execute('SELECT kind,content FROM artifact_descriptions WHERE artifact_id=%s AND language=%s ORDER BY kind',(artifact,'en')).fetchall()
        require(descriptions==[dict(kind='dejargonized',content=plain),dict(kind='technical',content=technical)],'Legacy descriptions differ')
        actual=db.execute('SELECT validity_statement,evidence_ref_key,review_status FROM artifact_validity_bounds WHERE version_id=%s',(item['version_id'],)).fetchall()
        require(actual==[dict(**bound,review_status='automated_pass')],'Legacy reviewed regime differs')
        actual=db.execute('SELECT ref_id,ref_key,title,url,source,verified,confidence,relevance_note FROM artifact_references WHERE artifact_id=%s AND ref_key=%s',(artifact,reference['ref_key'])).fetchall()
        require(actual==[reference],'Legacy scientific reference differs')
        selected.append((graph,record))
    return selected


def promote(source,apply=False):
    if apply:
        gates=json.loads((ROOT/'docs/reviews/legacy_physics_publication_transaction.json').read_text())
        require(gates['passed'] and gates['post_activation_failure_rolled_back'] and gates['publisher_sha256']==sha(__file__)
            and gates['validator_sha256']==sha(ROOT/'scripts/validate_legacy_physics_publication_transaction.py'),'Qualified publication rollback required')
    qualification=review(source);created=0;updated=0;already=[]
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c statement_timeout=30000 -c lock_timeout=10000') as db:
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(DRAFT_RUNNER,))
        def ensure(table,keys,row):
            nonlocal created
            created+=int(ensure_row(db,table,keys,row))
        for item in sorted(qualification['graphs'],key=lambda r:r['artifact_id']):
            artifact=UUID(item['artifact_id']);version=UUID(item['version_id'])
            state=db.execute('SELECT status,is_publishable,fqdn FROM artifacts WHERE artifact_id=%s FOR UPDATE',(artifact,)).fetchone()
            done=state['status']=='approved' and state['is_publishable'];already.append(done)
            graph,record=check(db,item['family'],done)
            technical,plain,bound,reference=publication_fields(db,item)
            ensure('artifact_references',dict(artifact_id=artifact,ref_key=reference['ref_key']),dict(artifact_id=artifact,**reference))
            for kind,content in [('technical',technical),('dejargonized',plain)]:
                ensure('artifact_descriptions',dict(artifact_id=artifact,kind=kind,language='en'),dict(description_id=uuid5(version,RUNNER+':'+kind),artifact_id=artifact,kind=kind,content=content,language='en',generated_by=RUNNER,reviewed=False))
            rollup=dict(overall_verdict='acceptable_with_limits',structural_status='pass',runtime_status='pass',semantic_status='pass',developer_semantics_status='pass',
                review_status='approved',review_semantic_verdict='pass',review_developer_semantics_verdict='pass',trust_readiness='ready',review_limitations=item['limitations'],
                review_required_actions=[],trust_blockers=[],acceptability_band='acceptable_with_limits',parity_coverage_level='positive_and_negative',parity_test_status='pass')
            ensure('artifact_audit_rollups',dict(artifact_id=artifact),dict(artifact_id=artifact,**rollup))
            evidence=uuid5(version,RUNNER)
            ensure('artifact_audit_evidence',dict(evidence_id=evidence),dict(evidence_id=evidence,artifact_id=artifact,version_id=version,audit_type='semantic_audit',passed=True,status='completed',source_kind='automated',runner_version=RUNNER,
                details=Jsonb(dict(qualification=qualification,publisher_sha256=sha(__file__),publication_tier=3))))
            bound_id=uuid5(version,RUNNER+':regime')
            ensure('artifact_validity_bounds',dict(bound_id=bound_id),dict(bound_id=bound_id,artifact_id=artifact,version_id=version,scope='version',bound_kind='regime',review_status='automated_pass',**bound,
                metadata=Jsonb(dict(approval_evidence_id=str(evidence),review_source='automated',qualified_parent_version_id=item['approved_family_qualification']['version_id'],scope='Inherited explicitly qualified family regime; no human certification.'))))
            if not done:
                topo=build_published_cdg_projection(artifact=dict(artifact_id=str(artifact),fqdn=state['fqdn']),version=dict(version_id=str(version),content_hash=item['graph_sha256']),cdg=graph).topo_hash
                updated+=db.execute("UPDATE artifacts SET status='approved',is_publishable=true,description=%s,verified_leaf_coverage=1,leaf_count=1,top_level_input_arity=%s,top_level_output_arity=%s,topo_hash=%s WHERE artifact_id=%s",(technical,len(graph.nodes[0].inputs),len(graph.nodes[0].outputs),topo,artifact)).rowcount
                updated+=db.execute('UPDATE artifact_versions SET is_latest=false WHERE artifact_id=%s AND is_latest',(artifact,)).rowcount
                updated+=db.execute('UPDATE artifact_versions SET is_latest=true WHERE version_id=%s AND NOT is_latest',(version,)).rowcount
        check_publication(db,qualification)
        require(review(source)==qualification,'Qualification changed during activation')
        if not apply:db.rollback()
    return dict(applied=apply,already_approved=all(already),rows_created=created,rows_updated=updated,trust_tier=3,
        graphs=[{k:r[k] for k in ['family','artifact_id','version_id','graph_sha256']} for r in qualification['graphs']],original_histories_preserved=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True);parser.add_argument('--apply',action='store_true')
    args=parser.parse_args();report=promote(args.source_directory,args.apply)
    if args.apply:
        name='legacy_physics_publication_repeat.json' if report['already_approved'] else 'legacy_physics_publication.json'
        (ROOT/'docs/reviews'/name).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='graphs'}))
