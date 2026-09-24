"""Review the stored radius revision for scoped automated Tier 3 publication."""
import json
from pathlib import Path

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from scripts.check_schwarzschild_revision import check
from scripts.stage_schwarzschild_original_revision import ROOT,ARTIFACT,EXECUTION,sha,require
from scripts.audit_schwarzschild_original_scope import audit
from scripts.validate_schwarzschild_revision_database_gates import cases

LIMITATIONS=[
    'Automated Tier 3 numerical length-scale realization under the source artifact identity; no Tier 1 human certification.',
    'Evaluate 2GM/c^2 for nonempty positive finite SI mass and positive finite scalar G and c. Output is positive finite float64 metres with the mass shape preserved.',
    'One approved dimensioned provider is reused. No dataset-specific implementation, fitted model or implicit physical constants are introduced.',
    'Five parsed original expressions reconcile with normalized source expressions under positive real assumptions. Four forward proof steps and seven conditions are verified; historical failed evidence remains unchanged.',
    '208 numerical values over five runner cases match a 100-digit Decimal reference within the qualified three-ULP tolerance. Positive subnormal outputs may have reduced relative precision.',
    'Horizon interpretation requires caller-established spherical nonrotating uncharged Schwarzschild regime with an asymptotically flat vacuum exterior.',
    'No black-hole classification, collapse simulation, rotating or charged horizon model, field-equation solution or certification of Newtonian light-motion premises.',
    'Provisioned in-process runtime and catalog metadata only; HTTP transport, clean installation and package redistribution remain unqualified.',
    'Current catalog state and version-bound semantic evidence govern selection; graph metadata preserves its historical draft provenance.',
]


def review(source):
    directory=ROOT/'docs/reviews'
    scope=audit(source)
    require(scope==json.loads((directory/'physics_schwarzschild_original_scope.json').read_text()),'Source scope changed')
    execution=json.loads((directory/'schwarzschild_revision_catalog_execution.json').read_text())
    gates=json.loads((directory/'schwarzschild_revision_database_gates.json').read_text())
    require(execution['passed'] and execution['catalog_mutations']==0 and execution['original_history_preserved'],'Stored execution required')
    numerical=execution['stored_execution']
    require(numerical['full_runner_cases']==5 and numerical['synthetic_numeric_cases']==208
        and numerical['maximum_ulp_error']<=numerical['ulp_tolerance']==3 and numerical['exact_rational_proof_cases']==3,'Numerical coverage differs')
    require(execution['validator_sha256']==sha(ROOT/'scripts/validate_schwarzschild_revision_execution.py'),'Execution validator changed')
    for name,digest in numerical['implementation_sha256'].items():require(sha(ROOT/name)==digest,'Numerical implementation changed')
    require(gates['passed'] and gates['rollback_verified'] and gates['committed_catalog_mutations']==0
        and gates['rejected_faults']==[c[0] for c in cases(execution['version_id'])],'Negative gates incomplete')
    require(gates['validator_sha256']==sha(ROOT/'scripts/validate_schwarzschild_revision_database_gates.py')
        and gates['checker_sha256']==sha(ROOT/'scripts/check_schwarzschild_revision.py'),'Integrity code changed')
    require(gates['graph_sha256']==execution['graph_sha256'] and gates['version_id']==execution['version_id'],'Evidence identity differs')
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        state=db.execute('SELECT status,is_publishable FROM artifacts WHERE artifact_id=%s',(ARTIFACT,)).fetchone()
        graph,imported=check(db,state==dict(status='approved',is_publishable=True))
        require(imported['version_id']==execution['version_id'] and imported['graph_sha256']==numerical['graph_digest'],'Stored qualification differs')
        rows=db.execute("SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='schwarzschild-community.v1' AND passed",(EXECUTION,)).fetchall()
        require(len(rows)==1,'Unique parent approval required')
        parent=rows[0]['details']
        require(parent['publication_tier']==3 and parent['review_source']=='automated','Parent tier differs')
        for name,digest in parent['evidence_sha256'].items():require(sha(directory/name)==digest,'Parent evidence changed')
    tests=json.loads((directory/'schwarzschild_test_review.json').read_text())
    require(tests['test_results']==dict(tests=33,failures=0,errors=0,skipped=0),'Complete provider test qualification required')
    require(tests['provider_sha256']==numerical['provider_sha256'],'Provider test identity differs')
    for name,digest in tests['test_source_sha256'].items():require(sha(ROOT/name)==digest,'Provider test code changed')
    names=['physics_schwarzschild_original_scope.json','schwarzschild_original_revision_transaction.json',
        'schwarzschild_original_revision_import.json','schwarzschild_original_revision_repeat.json',
        'schwarzschild_revision_catalog_execution.json','schwarzschild_revision_database_gates.json','schwarzschild_test_review.json','schwarzschild_semantic_review.md']
    return dict(eligible_for_approval_transaction=True,approved=False,proposed_tier=3,review_source='automated',catalog_mutations=0,
        artifact_id=ARTIFACT,version_id=execution['version_id'],graph_sha256=execution['graph_sha256'],reused_atoms=1,new_atoms=0,
        original_history_sha256=imported['original_history_sha256'],limitations=LIMITATIONS,
        evidence_sha256={name:sha(directory/name) for name in names},reviewer_sha256=sha(__file__))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    report=review(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/schwarzschild_revision_publication_review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(eligible=True,proposed_tier=3,approved=False)))
