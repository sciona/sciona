"""Require complete stored execution and preserved evidence for Tier 3 promotion."""
import json
import math
import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from scripts.plan_ensemble_corrected_intake import ROOT,plan,sha,require
from scripts.check_ensemble_corrected_intake import check_candidate

LIMITATIONS=[
    'Automated Tier 3 qualification of the corrected original intake; no human Tier 1 certification.',
    'All eleven source model branches remain explicit in 155 nodes, with 59 approved reusable provider versions and 26 registered dependency links.',
    'Three ordered populations and separate caller-supplied training memberships per model family are required. Caller establishes label meaning, safe membership and common physical prediction identity.',
    'Source configurations remain fixed. Alex consumes channels-by-samples clips at 400 Hz; Feng consumes samples-by-channels source-duration clips. Andriy requires explicit CSP candidates, sequence order, primary/auxiliary positive and negative groups, and sampling frequencies.',
    'Source finite-input, feature-selection and nondegenerate CSP requirements remain enforced. Generic list root types do not imply arbitrary input compatibility; downstream provider contracts apply.',
    'Autocorrelation uses the identity tangent reference without prediction updates. Coherence uses the log-Euclidean reference with whole-prediction-batch updates; changing batch composition can change predictions.',
    'The source iterative affine-invariant-mean and global distance-isometry claims are not certified. Alex maximum aggregation is not imposed on other families.',
    'Full-vector ranks precede identity selection. Eleven equal weights add to the supplied baseline without clipping or reranking. Extra identities affect ranks; missing and duplicate identities are rejected.',
    'Reusable numerical providers can be selected independently for other disciplines when their contracts hold. This fixed source composition does not establish predictive effectiveness in a new domain.',
    'Synthetic raw execution and pinned independent component references qualify computational behavior. They do not establish historical-engine equivalence, real-world accuracy or clinical utility.',
    'Rebuilt R libraries have separately qualified binary identities. Historical model evidence remains unchanged; the original binary hashes are not claimed.',
    'The original intake version, snapshot and evidence remain preserved. Approval selects the corrected execution version, not the original placeholder graph.',
    'Execution requires the qualified Python and external R runtime. Clean installation, hosted deployment and binary redistribution are outside this qualification.',
]
FAULTS={'approval','trust_tier','lineage','content_hash','root_contract','binding_runtime',
    'binding_version','binding_alias','provenance','staging_evidence','failed_audit',
    'node_projection','edge_projection','original_selection','provider_tier'}


def review():
    proposed=plan();directory=ROOT/'docs/reviews'
    execution=json.loads((directory/'ensemble_corrected_intake_execution.json').read_text())
    require(execution.get('passed') is True and execution.get('stored_candidate') is True
        and execution.get('approved') is False and execution.get('catalog_mutations')==0,'Completed stored candidate evidence required')
    require(execution['candidate_version_id']==proposed['version_id'] and execution['original_artifact_id']==proposed['artifact_id']
        and execution['graph_digest']==proposed['graph_sha256'],'Executed candidate identity differs')
    require(execution['nodes']==execution['executed_nodes']==155 and execution['edges']==523
        and execution['full_runner_cases']==1 and execution['populations']==3 and execution['prediction_segments']==6
        and execution['synthetic_only'] and execution['exact_final_source_blend'] and execution['native_threads']==1,'Execution coverage differs')
    spreads=execution['model_score_spreads']
    require(len(spreads)==11 and all(math.isfinite(v) and v>0 for v in spreads)
        and math.isfinite(execution['final_score_spread']) and execution['final_score_spread']>0,'Nondegenerate complete model outputs required')
    require(execution['provider_sha256']=={b['runtime_fqdn']:b['provider_sha256'] for b in proposed['bindings']},'Executed provider set differs')
    for field,path in [('wrapper_sha256','scripts/validate_ensemble_corrected_intake_execution.py'),
        ('validator_sha256','scripts/validate_ensemble_execution.py'),('candidate_checker_sha256','scripts/check_ensemble_corrected_intake.py'),
        ('graph_source_sha256','sciona/ensemble_execution.py'),('plan_sha256','docs/reviews/ensemble_corrected_intake_plan.json'),
        ('restored_models_sha256','docs/reviews/ensemble_restored_r_models.json')]:
        require(execution[field]==sha(ROOT/path),'Execution evidence/code changed: '+field)
    gates=json.loads((directory/'ensemble_corrected_intake_integrity.json').read_text())
    require(gates['passed'] and gates['committed_catalog_mutations']==0 and set(gates['cases'])==FAULTS
        and all(c['rejected'] for c in gates['cases'].values()),'Complete corruption rejection required')
    require(gates['candidate_version_id']==proposed['version_id'] and gates['graph_sha256']==proposed['graph_sha256']
        and gates['checker_sha256']==execution['candidate_checker_sha256'] and gates['plan_sha256']==execution['plan_sha256']
        and gates['validator_sha256']==sha(ROOT/'scripts/validate_ensemble_corrected_intake_integrity.py'),'Corruption evidence changed')
    transaction=json.loads((directory/'ensemble_corrected_intake_draft_transaction.json').read_text())
    require(transaction['passed'] and transaction['failure_rollback_verified'] and transaction['committed_catalog_mutations']==0
        and transaction['injected_after_actual_writes']==[3,428,856]
        and transaction['rollback']['version_id']==proposed['version_id']
        and transaction['importer_sha256']==sha(ROOT/'scripts/import_ensemble_corrected_intake.py')
        and transaction['validator_sha256']==sha(ROOT/'scripts/validate_ensemble_corrected_intake_draft_transaction.py'),'Staging transaction evidence differs')
    tests=json.loads((directory/'ensemble_test_review.json').read_text())
    require(tests['test_results']==dict(tests=24,failures=0,errors=0,skipped=0),'Retained contract coverage differs')
    for name,digest in tests['test_source_sha256'].items():
        require(sha(ROOT/'tests'/name)==digest,'Retained tests changed')
    providers={a['runtime_fqdn']:a['provider_sha256'] for a in proposed['atoms']}
    require(all(providers[name]==digest for name,digest in tests['adapter_source_sha256'].items()),'Tested adapters changed')
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        row=db.execute('SELECT status,is_publishable FROM artifacts WHERE artifact_id=%s',(proposed['artifact_id'],)).fetchone()
        check_candidate(db,proposed,approved=row==dict(status='approved',is_publishable=True))
    names=['ensemble_corrected_intake_execution.json','ensemble_corrected_intake_integrity.json',
        'ensemble_corrected_intake_plan.json','ensemble_corrected_intake_draft_transaction.json',
        'ensemble_corrected_intake_draft_import.json','ensemble_corrected_intake_draft_repeat.json',
        'ensemble_test_review.json','ensemble_alignment_parity.json','ensemble_semantic_review.md',
        *proposed['evidence_sha256']]
    return dict(format='ensemble-corrected-intake-publication-review.v1',review_source='automated',
        proposed_tier=3,eligible_for_approval_transaction=True,approved=False,catalog_mutations=0,
        artifact_id=proposed['artifact_id'],version_id=proposed['version_id'],graph_sha256=proposed['graph_sha256'],
        original_version_id=proposed['original_version_id'],original_content_hash=proposed['original_content_hash'],
        original_rows_sha256=proposed['original_rows_sha256'],new_atoms=0,reused_approved_atoms=59,
        scope=proposed['scope'],limitations=LIMITATIONS,evidence_sha256={name:sha(directory/name) for name in names},
        planner_sha256=proposed['planner_sha256'],reviewer_sha256=sha(__file__))


if __name__=='__main__':
    result=review()
    (ROOT/'docs/reviews/ensemble_corrected_intake_publication_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(eligible=True,proposed_tier=3,approved=False)))
