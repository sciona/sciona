"""Review a corrected original intake without draft-only historical audit assumptions."""
import json
from pathlib import Path
import tomllib

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from scripts.plan_nasa_corrected_intake import ROOT, plan, require, sha
from scripts.check_nasa_corrected_intake import check_candidate
from scripts.validate_nasa_corrected_intake_database_gates import fault_cases
from scripts.review_conditional_correction_environment import dependency_closure
from scripts.review_residual_classifier_publication import review_notices

LIMITATIONS=[
    'Automated Tier 3 corrected executable revision under the original intake identity; no Tier 1 human certification or Tier 2 empirical-effectiveness claim.',
    'This entrypoint requires ten labeled training populations and ten queried populations. Separately approved lifecycle graphs support sparse inference and reuse of saved models.',
    'All 485 execution nodes bind directly to 39 existing approved atoms. Shared numerical and keyed-routing atoms remain domain independent; airport feature, partition and identity contracts are explicit adapters.',
    'The original version, source snapshot and graph records remain immutable. Current selection advances to the corrected executable version; the old five-stage placeholder graph is not certified as executable.',
    'Corrected feature construction uses epoch time, departure estimates, airline categories and observation-available taxi-to-gate history. The unsupported original weather-severity claim is excluded.',
    'Source corrections include complete group coverage, deterministic vocabulary, explicit observation cutoffs, grouped residual selection, rounded classifier training features and unrounded query features.',
    'The complete graph trains before handing off state, then performs no fitting during inference. Conditional median offsets use strict probability branches, float32 arithmetic and checked int32 minute outputs.',
    'Whole-graph static table propagation remains unsupported. Runtime checks validate materialized numerical boundaries before fitting.',
    '640 exact synthetic predictions qualify this stored version. Another 128 sparse candidate predictions are supplementary regression evidence, not approval of a second complete-workflow entrypoint.',
    'Synthetic execution establishes computational behavior and reuse, not empirical predictive effectiveness or historical backend identity.',
    'Caller-supplied records and complete runtime model state replace source filesystem/notebook orchestration. Non-public inputs, derived tables and learned state remain private runtime material.',
    'Provisioned in-process execution is qualified with optional server accelerators excluded. HTTP deployment, clean installation and binary/package redistribution remain unqualified.',
    'Native state is tied to its exact backend version. Integrity digests detect corruption and do not establish authenticity.',
]


def review():
    proposed=plan(); directory=ROOT/'docs/reviews'
    names=['nasa_corrected_intake_catalog_execution.json','nasa_corrected_intake_runtime_profile.json',
        'nasa_corrected_intake_database_gates.json','nasa_population_publication_review.json']
    execution,runtime,gates,parent=[json.loads((directory/name).read_text()) for name in names]
    require(execution['passed'] and runtime['passed'] and runtime['execution']==execution,'Matching runtime evidence required')
    require(runtime['catalog_evidence_sha256']==sha(directory/names[0]),'Catalog evidence hash differs')
    require(execution['stored_predictions_compared']==640 and execution['additional_candidate_predictions']==128
        and execution['reused_atoms']==39 and execution['bindings']==485,'Execution coverage differs')
    for report,filename in [(execution,'validate_nasa_corrected_intake_catalog.py'),
            (runtime,'validate_nasa_corrected_intake_runtime_profile.py'),(gates,'validate_nasa_corrected_intake_database_gates.py')]:
        require(report['validator_sha256']==sha(ROOT/'scripts'/filename),'Qualification code drift: '+filename)
    for report in [execution,gates]:
        require(report['graph_sha256']==proposed['graph_sha256'] and report['version_id']==proposed['version_id']
            and report['artifact_id']==proposed['artifact_id'],'Qualification identity differs')
        require(report['checker_sha256']==sha(ROOT/'scripts/check_nasa_corrected_intake.py')
            and report['planner_sha256']==proposed['planner_sha256'],'Qualification checker/planner drift')
    require(gates['passed'] and gates['rollback_verified'] and gates['committed_catalog_mutations']==0
        and gates['rejected_faults']==[case[0] for case in fault_cases(proposed)],'Database fault coverage differs')
    require(gates['original_rows_sha256']==proposed['original_rows_sha256'],'Original history evidence differs')
    require(runtime['guard_source_sha256']==sha(ROOT/'scripts/validate_residual_classifier_runtime_profile.py'),'Import guard changed')
    require(runtime['excluded_optional_packages']==['httptools','psycopg_binary','uvloop','watchfiles']
        and runtime['psycopg_implementation']=='python' and runtime['libpq_version']==psycopg.pq.version(),'Runtime profile differs')
    manifests={'sciona':ROOT/'pyproject.toml','sciona-atoms-ml':ROOT.parent/'sciona-atoms-ml/pyproject.toml'}
    require(runtime['manifest_sha256']=={name:sha(path) for name,path in manifests.items()},'Manifest drift')
    projects={name:tomllib.loads(path.read_text())['project'] for name,path in manifests.items()}
    closure=dependency_closure(['sciona-atoms-ml[xgboost]','sciona','psycopg','python-dotenv'],projects)
    require(closure['compatible'] and runtime['dependency_closure']['compatible']
        and closure['versions']==runtime['dependency_closure']['versions']==parent['dependency_versions'],'Dependency closure drift')
    notices=review_notices(directory)
    require(notices==parent['notice_review'],'Retained notice review differs')
    # Verify frozen parent evidence, without rerunning its superseded draft-only
    # source-state assumption. plan() checks current parent graphs and atoms.
    require(parent['eligible_for_approval_transaction'] and parent['proposed_tier']==3,'Parent qualification required')
    for name,digest in parent['evidence_sha256'].items():
        require(sha(directory/name)==digest,'Frozen parent evidence changed: '+name)
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        state=db.execute('SELECT status,is_publishable FROM artifacts WHERE artifact_id=%s',(proposed['artifact_id'],)).fetchone()
        approved=state==dict(status='approved',is_publishable=True)
        check_candidate(db,proposed,approved=approved)
    names+=['nasa_original_intake_scope_audit.json','nasa_complete_workflow_execution.json',
        'nasa_corrected_intake_draft_transaction.json','nasa_corrected_intake_draft_import.json',
        'residual_classifier_dependency_notices.json','residual_classifier_tokenizers_notice.json']
    return dict(format='nasa-corrected-intake-publication-review.v1',review_source='automated',proposed_tier=3,
        eligible_for_approval_transaction=True,approved=False,catalog_mutations=0,
        artifact_id=proposed['artifact_id'],version_id=proposed['version_id'],graph_sha256=proposed['graph_sha256'],
        original_version_id=proposed['original_version_id'],original_content_hash=proposed['original_content_hash'],
        original_rows_sha256=proposed['original_rows_sha256'],new_atoms=0,reused_approved_atoms=39,
        scope=proposed['scope'],provenance_policy=proposed['provenance_policy'],limitations=LIMITATIONS,
        corrected_descriptions=proposed['corrected_descriptions'],dependency_versions=closure['versions'],notice_review=notices,
        evidence_sha256={name:sha(directory/name) for name in names},
        planner_sha256=proposed['planner_sha256'],reviewer_sha256=sha(Path(__file__)))


if __name__=='__main__':
    report=review()
    (ROOT/'docs/reviews/nasa_corrected_intake_publication_review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(eligible=True,proposed_tier=3,new_atoms=0,reused_atoms=39,approved=False)))
