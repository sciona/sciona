"""Review exact legacy executions without rewriting historical source approvals."""
import argparse
import importlib
import json
from pathlib import Path
import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from scripts.stage_legacy_physics_revisions import ROOT,sha,require
from scripts.check_legacy_physics_revision import check
from scripts.validate_legacy_physics_database_gates import cases


def review(source):
    directory=ROOT/'docs/reviews'
    scope=json.loads((directory/'legacy_physics_execution_scope.json').read_text())
    imported=json.loads((directory/'legacy_physics_revision_import.json').read_text())
    execution=json.loads((directory/'legacy_physics_catalog_execution.json').read_text())
    gates=json.loads((directory/'legacy_physics_database_gates.json').read_text())
    require(execution['passed'] and execution['catalog_mutations']==0 and gates['passed']
        and gates['rollback_verified'] and gates['committed_catalog_mutations']==0,'Execution and rollback gates required')
    require(execution['validator_sha256']==sha(ROOT/'scripts/validate_legacy_physics_execution.py')
        and gates['validator_sha256']==sha(ROOT/'scripts/validate_legacy_physics_database_gates.py')
        and gates['checker_sha256']==execution['checker_sha256']==sha(ROOT/'scripts/check_legacy_physics_revision.py'),'Legacy validation code changed')
    families={r['family'] for r in scope['graphs']}
    for report in [imported,execution,gates]:
        require(len(report['graphs'])==len(families)==5 and {r['family'] for r in report['graphs']}==families,'Complete five-family evidence required')
    reviews=[]
    for item in scope['graphs']:
        family=item['family'];parent_reviewer=importlib.import_module('scripts.review_'+family+'_revision')
        qualification=parent_reviewer.review(source)
        require(qualification==item['qualification'],'Fresh approved-family qualification differs')
        record=next(r for r in imported['graphs'] if r['family']==family)
        numerical=next(r for r in execution['graphs'] if r['family']==family)
        negative=next(r for r in gates['graphs'] if r['family']==family)
        for report in [numerical,negative]:
            require(report['version_id']==record['version_id'] and report['graph_sha256']==record['graph_sha256'],'Evidence version differs')
        require(negative['rejected_faults']==[c[0] for c in cases(record)],'Legacy negative gates incomplete')
        oracle=ROOT/('scripts/validate_'+family+('_revision_execution.py' if family in ['series','period_frequency'] else '_execution.py'))
        require(numerical['oracle_sha256']==sha(oracle),'Family oracle changed')
        result=numerical['execution']
        for key in ['implementation_sha256','source_sha256']:
            for path,digest in result.get(key,{}).items():require(sha(ROOT/path)==digest,'Family execution dependency changed')
        if family=='quadratic':
            require(result['synthetic_polynomials']==305 and result['full_runner_cases']==5 and result['maximum_ulp_error']<=3,'Quadratic coverage differs')
            reuse=numerical['reuse'];require(reuse['passed'] and reuse['distinct_synthetic_cases']==256 and reuse['full_runner_cases']==4
                and reuse['graph_sha256']==record['graph_sha256'] and reuse['validator_sha256']==sha(ROOT/'scripts/validate_quadratic_revision_reuse.py'),'Quadratic reuse differs')
        elif family=='euler':
            require(result['full_runner_cases']==2 and result['proof_steps_per_run']==4 and result['exact_source_interpreted_step_parity'],'Euler proof coverage differs')
        elif family=='schwarzschild':
            require(result['full_runner_cases']==5 and result['synthetic_numeric_cases']==208 and result['maximum_ulp_error']<=3,'Radius coverage differs')
        elif family=='series':
            require(result['passed'] and result['independent_circuit_cases']==256 and len(result['invalid_cases_rejected'])==10,'Series coverage differs')
        else:
            require(result['passed'] and result['independent_measurement_cases']==256 and len(result['invalid_cases_rejected'])==9,'Frequency coverage differs')
        with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
                options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
            state=db.execute('SELECT status,is_publishable FROM artifacts WHERE artifact_id=%s',(record['artifact_id'],)).fetchone()
            check(db,family,state==dict(status='approved',is_publishable=True))
        reviews.append(dict(family=family,artifact_id=record['artifact_id'],version_id=record['version_id'],graph_sha256=record['graph_sha256'],
            source_scope_sha256=item['qualification_sha256'],approved_family_qualification=qualification,
            original_history_sha256=record['original_history_sha256'],limitations=qualification['limitations'],reused_atoms=1,new_atoms=0))
    names=['legacy_physics_execution_scope.json','legacy_physics_revision_transaction.json','legacy_physics_revision_import.json',
        'legacy_physics_revision_repeat.json','legacy_physics_catalog_execution.json','legacy_physics_database_gates.json']
    return dict(eligible_for_approval_transaction=True,approved=False,proposed_tier=3,review_source='automated',catalog_mutations=0,graphs=reviews,
        limitations=['Legacy split-node histories are preserved as provenance; approval concerns the explicitly staged executable revisions only.',
            'All inherited family corrections, source-interpretation and application limits remain binding.',
            'Provisioned in-process catalog execution only; no HTTP/search ranking, clean-install or redistribution qualification.'],
        evidence_sha256={name:sha(directory/name) for name in names},reviewer_sha256=sha(__file__))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    report=review(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/legacy_physics_publication_review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(eligible=True,proposed_tier=3,graphs=5,approved=False)))
