"""Atomically activate the qualified revision of the original intake at Tier 3."""
import argparse
import json
from pathlib import Path
from uuid import UUID, uuid5

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from sciona.cdg_projection import build_published_cdg_projection
from sciona.nasa_complete_workflow_graph import build_complete_workflow_graph
from scripts.import_residual_execution_drafts import ensure_row
from scripts.plan_nasa_corrected_intake import ROOT, plan, require, sha
from scripts.check_nasa_corrected_intake import check_candidate, PUBLICATION_RUNNER
from scripts.review_nasa_corrected_intake_publication import review, LIMITATIONS

RUNNER=PUBLICATION_RUNNER
DESCRIPTION=('Corrected ten-population training and prediction workflow: deterministic airport feature adapters, '
    'grouped internal regression, residual-based retained-population regression, underestimation classification, '
    'conditional median calibration and identity-aligned int32 minute predictions. '
    'Composes 39 approved reusable numerical, state, routing and domain-adapter atoms in 485 explicit nodes. '
    'Automated Tier 3 synthetic execution evidence; empirical effectiveness remains unqualified.')
PLAIN_DESCRIPTION=('Train models for ten airport populations, then predict for those ten populations using explicit '
    'time, departure-estimate, airline-category and taxi-to-gate features. Calibrate predictions from held-out errors '
    'and preserve input row identity. Reusable numerical operations are separate from airport adapters. '
    'This corrected executable version preserves the original source history.')


def promote(apply=False):
    if apply:
        gates=json.loads((ROOT/'docs/reviews/nasa_corrected_intake_publication_transaction_gates.json').read_text())
        require(gates['passed'] and gates['publisher_sha256']==sha(Path(__file__))
            and gates['validator_sha256']==sha(ROOT/'scripts/validate_nasa_corrected_intake_publication_transaction.py')
            and gates['injected_transaction_failure_rolled_back'],'Publication transaction qualification required')
    qualification,proposed=review(),plan()
    graph=build_complete_workflow_graph(10)
    topology=build_published_cdg_projection(artifact={'artifact_id':proposed['artifact_id'],'fqdn':proposed['fqdn']},
        version={'version_id':proposed['version_id'],'content_hash':proposed['graph_sha256']},cdg=graph).topo_hash
    created=0;updated=0
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c statement_timeout=30000 -c lock_timeout=10000') as db:
        for name in ['residual-classifier-draft.v1','nasa-domain-draft.v1','nasa-lifecycle-draft.v1','nasa-population-draft.v1','nasa-corrected-intake-draft.v1']:
            db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(name,))
        state=db.execute('SELECT status,is_publishable FROM artifacts WHERE artifact_id=%s FOR UPDATE',(proposed['artifact_id'],)).fetchone()
        already=state==dict(status='approved',is_publishable=True)
        check_candidate(db,proposed,approved=already)
        def ensure(table,keys,row):
            nonlocal created
            created+=int(ensure_row(db,table,keys,row))
        target=UUID(proposed['artifact_id']);version=UUID(proposed['version_id'])
        for kind,content in [('technical',DESCRIPTION),('dejargonized',PLAIN_DESCRIPTION)]:
            ensure('artifact_descriptions',dict(artifact_id=target,kind=kind,language='en'),
                dict(description_id=uuid5(version,RUNNER+':'+kind),artifact_id=target,kind=kind,
                    content=content,language='en',generated_by=RUNNER,reviewed=False))
        rollup=dict(overall_verdict='acceptable_with_limits',structural_status='pass',runtime_status='pass',
            semantic_status='pass',developer_semantics_status='pass',review_status='approved',review_semantic_verdict='pass',
            review_developer_semantics_verdict='pass',trust_readiness='ready',review_limitations=LIMITATIONS,
            review_required_actions=[],trust_blockers=[],acceptability_band='acceptable_with_limits',
            parity_coverage_level='positive_and_negative',parity_test_status='pass')
        ensure('artifact_audit_rollups',dict(artifact_id=target),dict(artifact_id=target,**rollup))
        evidence=uuid5(version,RUNNER)
        ensure('artifact_audit_evidence',dict(evidence_id=evidence),dict(evidence_id=evidence,artifact_id=target,
            version_id=version,audit_type='semantic_audit',passed=True,status='completed',source_kind='automated',
            runner_version=RUNNER,details=Jsonb(dict(qualification=qualification,publication_tier=3,
                publisher_sha256=sha(Path(__file__)),description=DESCRIPTION,plain_description=PLAIN_DESCRIPTION))))
        if not already:
            updated+=db.execute("UPDATE artifacts SET status='approved',is_publishable=true,description=%s,verified_leaf_coverage=1,"
                'leaf_count=%s,top_level_input_arity=%s,top_level_output_arity=%s,topo_hash=%s WHERE artifact_id=%s',
                (DESCRIPTION,len(graph.nodes),len(proposed['inputs']),len(proposed['outputs']),topology,target)).rowcount
            # Clear the old latest flag before activation, including databases
            # enforcing a unique latest version per artifact.
            updated+=db.execute('UPDATE artifact_versions SET is_latest=false WHERE artifact_id=%s AND is_latest',(target,)).rowcount
            updated+=db.execute('UPDATE artifact_versions SET is_latest=true WHERE version_id=%s AND NOT is_latest',(version,)).rowcount
        require(db.execute('SELECT description,topo_hash,leaf_count FROM artifacts WHERE artifact_id=%s',(target,)).fetchone()
            ==dict(description=DESCRIPTION,topo_hash=topology,leaf_count=len(graph.nodes)),'Published artifact metadata differs')
        check_candidate(db,proposed,approved=True)
        selected=db.execute('SELECT version_id FROM artifact_versions WHERE artifact_id=%s AND is_latest',(target,)).fetchall()
        require(selected==[dict(version_id=version)],'Corrected latest selection required')
        require(review()==qualification,'Qualification changed during transaction')
        if not apply:db.rollback()
    return dict(applied=apply,already_approved=already,rows_created=created,rows_updated=updated,trust_tier=3,
        artifact_id=proposed['artifact_id'],version_id=proposed['version_id'],graph_sha256=proposed['graph_sha256'],
        original_records_preserved=True,served_cdgs_in_transaction=1,new_atoms=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true')
    report=promote(parser.parse_args().apply)
    if report['applied']:
        name='nasa_corrected_intake_publication_repeat.json' if report['already_approved'] else 'nasa_corrected_intake_publication.json'
        (ROOT/'docs/reviews'/name).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
