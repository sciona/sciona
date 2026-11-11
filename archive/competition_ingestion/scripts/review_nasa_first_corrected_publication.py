"""Automated Tier 3 review of the complete corrected first-place intake."""
import json
from pathlib import Path

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from scripts.plan_nasa_first_corrected_intake import ROOT,plan,sha,require
from scripts.check_nasa_first_corrected_intake import check_candidate
from scripts.review_nasa_first_lifecycle_publication import review as provider_review

LIMITATIONS=[
    'Automated Tier 3 corrected executable revision; no Tier 1 human certification or empirical Tier 2 claim.',
    'This entrypoint trains all ten source populations and predicts one selected population per call.',
    'The 343-node lifecycle uses 47 approved dependencies including explicit nested branches and controller delivery. Generic operations remain separate from source-specific adapters.',
    'Twenty-one native fits were qualified separately. Complete stored traversal compares every fit operand before replaying those checkpoint models; no second full native training run is claimed.',
    'Raw preparation, score/importance computation, policy-bound serialization, restoration and native predictions execute normally in complete traversal.',
    'Training-time feature order, fills, vocabularies and calendar policy are bound to model state. Policy drift and state corruption fail before prediction fallbacks.',
    'Primary predictions are integer minutes; baseline fallback preserves fractional values. Constant and empty-query behavior retain explicit source policy.',
    'Corrected observation availability and query-batch independence require retraining; original empirical configuration and original weights are unqualified.',
    'The historical source version and failed callable preflight remain unchanged; only the corrected candidate becomes selectable.',
    'Synthetic cross-domain checks qualify computational reuse, not predictive effectiveness or historical backend identity.',
    'Provisioned in-process CPU profile only; clean installation, HTTP deployment, static symbolic propagation and binary redistribution are unqualified.',
    'Non-public records, policies, derived tables and model payloads remain private runtime state. Integrity hashes are not authentication.',
]


def review():
    proposed=plan();directory=ROOT/'docs/reviews'
    names=['competition_nasa_first_corrected_catalog_execution.json','competition_nasa_first_corrected_database_gates.json',
           'competition_nasa_first_corrected_intake_draft_transaction.json','competition_nasa_first_corrected_intake_draft_import.json']
    execution,gates,transaction,staged=[json.loads((directory/name).read_text()) for name in names]
    require(execution['passed'] and gates['passed'] and transaction['passed'] and staged['applied'],'Qualification incomplete')
    for report in [execution,gates]:
        require(report['version_id']==proposed['version_id'] and report['graph_sha256']==proposed['graph_sha256'],'Qualification version differs')
        require(report['plan_sha256']==sha(directory/'competition_nasa_first_corrected_intake_plan.json'),'Qualification plan differs')
        for name,digest in report['implementation_sha256'].items():require(sha(ROOT/name)==digest,'Qualification implementation changed')
    require(execution['version_selected_production_converter'] and execution['original_history_preserved'],'Stored execution provenance missing')
    result=execution['execution']
    require(result['executed_outer_nodes']==343 and result['exact_replayed_fit_operands']==21
        and result['new_native_fits']==0 and result['exact_native_prediction_comparisons']==64,'Stored traversal scope differs')
    for name,digest in result['implementation_sha256'].items():require(sha(ROOT/name)==digest,'Complete executor evidence changed')
    expected=['version_hash','version_tier','original_latest','publishability','root_contract','output_contract','node_binding',
        'inference_binding','runtime_binding','nested_evidence','output_evidence','failed_evidence','historical_failure','graph_edge']
    expected += [f'dependency_{i}' for i in range(47)]
    require(gates['rejected_faults']==expected and gates['rollback_verified'],'Original-workflow corruption coverage differs')
    require(transaction['failure_rollback_verified'] and transaction['injected_after_actual_writes']==[3,1120,2240]
        and transaction['importer_sha256']==sha(ROOT/'scripts/import_nasa_first_corrected_intake.py')
        and transaction['validator_sha256']==sha(ROOT/'scripts/validate_nasa_first_corrected_intake_draft_transaction.py'),'Draft rollback qualification differs')
    providers=provider_review()
    require(providers['eligible_for_approval_transaction'] and len(providers['provider_versions'])==35,'Lifecycle provider qualification differs')
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
                        options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        state=db.execute('SELECT status,is_publishable FROM artifacts WHERE artifact_id=%s',(proposed['artifact_id'],)).fetchone()
        check_candidate(db,proposed,approved=state==dict(status='approved',is_publishable=True))
    return dict(eligible_for_approval_transaction=True,review_source='automated',proposed_tier=3,approved=False,catalog_mutations=0,
        artifact_id=proposed['artifact_id'],version_id=proposed['version_id'],graph_sha256=proposed['graph_sha256'],
        original_rows_sha256=proposed['original_rows_sha256'],reused_approved_providers=47,new_providers=0,
        scope=proposed['scope'],limitations=LIMITATIONS,provider_review=providers,
        evidence_sha256={name:sha(directory/name) for name in names},planner_sha256=proposed['planner_sha256'],
        reviewer_sha256=sha(Path(__file__)))


if __name__=='__main__':
    report=review()
    (ROOT/'docs/reviews/competition_nasa_first_corrected_publication_review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(eligible=True,tier=3,reused_providers=47,approved=False)))
