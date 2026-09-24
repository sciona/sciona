"""Review the stored corrected quadratic revision for scoped automated Tier 3 publication."""
import json
from pathlib import Path

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from scripts.check_quadratic_revision import check
from scripts.stage_quadratic_original_revision import ROOT,ARTIFACT,EXECUTION,sha,require
from scripts.audit_quadratic_original_scope import audit
from scripts.validate_quadratic_revision_database_gates import cases

LIMITATIONS=[
    'Automated Tier 3 corrected quadratic realization under the original artifact identity; no Tier 1 human certification.',
    'Finite real float64 coefficients with identical nonempty shapes, nonzero a, nonnegative discriminant and finite representable roots.',
    'Both ordered real roots are returned for either sign of a. Repeated and zero roots are retained. No linear fallback or complex-root extension.',
    'One existing approved provider is reused. Unit consistency, physical applicability and selection of an admissible root are caller responsibilities.',
    'Original and replay histories remain unchanged. Three original/source discrepancies and four source arithmetic counterexamples remain explicit; no source-parity claim.',
    'Corrected proof retains b/a, both alternative branches and abs(a); symbolic factorization establishes completeness under the stated real domain.',
    '305 numerical polynomials match an independent 800-digit Decimal reference within three ULP. This is bounded numerical evidence, not universal correct rounding.',
    '256 synthetic mechanics and cost-model examples verify both roots, independent model residuals and signed coefficient rescaling. This is constructed transfer evidence, not empirical effectiveness.',
    'Provisioned in-process runtime and catalog metadata only; HTTP transport, clean installation and package redistribution remain unqualified.',
    'Current catalog state and version-bound semantic evidence govern selection; graph metadata preserves historical draft provenance.',
]


def review(source):
    directory=ROOT/'docs/reviews'
    scope=audit(source)
    require(scope==json.loads((directory/'physics_quadratic_original_scope.json').read_text()),'Source scope changed')
    execution=json.loads((directory/'quadratic_revision_catalog_execution.json').read_text())
    gates=json.loads((directory/'quadratic_revision_database_gates.json').read_text())
    require(execution['passed'] and execution['catalog_mutations']==0 and execution['original_history_preserved'],'Stored execution required')
    numerical=execution['stored_execution']
    require(numerical['full_runner_cases']==5 and numerical['synthetic_polynomials']==305
        and numerical['maximum_ulp_error']<=3 and numerical['corrected_proof']==scope['corrected_proof'],'Numerical coverage differs')
    require(execution['validator_sha256']==sha(ROOT/'scripts/validate_quadratic_revision_execution.py'),'Execution validator changed')
    for name,digest in numerical['implementation_sha256'].items():require(sha(ROOT/name)==digest,'Numerical implementation changed')
    require(gates['passed'] and gates['rollback_verified'] and gates['committed_catalog_mutations']==0
        and gates['rejected_faults']==[c[0] for c in cases(execution['version_id'])],'Negative gates incomplete')
    require(gates['validator_sha256']==sha(ROOT/'scripts/validate_quadratic_revision_database_gates.py')
        and gates['checker_sha256']==sha(ROOT/'scripts/check_quadratic_revision.py'),'Integrity code changed')
    require(gates['graph_sha256']==execution['graph_sha256'] and gates['version_id']==execution['version_id'],'Evidence identity differs')
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        state=db.execute('SELECT status,is_publishable FROM artifacts WHERE artifact_id=%s',(ARTIFACT,)).fetchone()
        graph,imported=check(db,state==dict(status='approved',is_publishable=True))
        require(imported['version_id']==execution['version_id'] and imported['graph_sha256']==numerical['graph_digest'],'Stored qualification differs')
        rows=db.execute("SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='quadratic-corrected-community.v1' AND passed",(EXECUTION,)).fetchall()
        require(len(rows)==1,'Unique parent approval required')
        parent=rows[0]['details']
        require(parent['publication_tier']==3 and parent['review_source']=='automated','Parent tier differs')
        for name,digest in parent['evidence_sha256'].items():require(sha(directory/name)==digest,'Parent evidence changed')
    tests=json.loads((directory/'quadratic_test_review.json').read_text())
    require(tests['test_results']==dict(tests=31,failures=0,errors=0,skipped=0),'Complete provider test qualification required')
    require(tests['provider_sha256']==numerical['provider_sha256'],'Provider test identity differs')
    for name,digest in tests['test_source_sha256'].items():require(sha(ROOT/name)==digest,'Provider test code changed')
    reuse=json.loads((directory/'quadratic_revision_reuse.json').read_text())
    require(reuse['passed'] and reuse['catalog_mutations']==0 and reuse['distinct_synthetic_cases']==256
        and reuse['full_runner_cases']==4 and reuse['version_id']==execution['version_id']
        and reuse['graph_sha256']==execution['graph_sha256'],'Reuse evidence incomplete')
    require({d['domain'] for d in reuse['domains']}=={'constant_acceleration','quadratic_revenue_linear_cost'}
        and all(d['both_roots_exact'] and d['independent_model_residuals_exact'] and d['signed_coefficient_rescaling_verified'] for d in reuse['domains']),'Reuse checks differ')
    require(reuse['validator_sha256']==sha(ROOT/'scripts/validate_quadratic_revision_reuse.py')
        and reuse['checker_sha256']==sha(ROOT/'scripts/check_quadratic_revision.py')
        and reuse['runner_sha256']==sha(ROOT/'sciona/visualizer/runner.py'),'Reuse implementation changed')
    names=['physics_quadratic_original_scope.json','quadratic_original_revision_transaction.json',
        'quadratic_original_revision_import.json','quadratic_original_revision_repeat.json',
        'quadratic_revision_catalog_execution.json','quadratic_revision_reuse.json','quadratic_revision_database_gates.json','quadratic_test_review.json','quadratic_correction_review.md']
    return dict(eligible_for_approval_transaction=True,approved=False,proposed_tier=3,review_source='automated',catalog_mutations=0,
        artifact_id=ARTIFACT,version_id=execution['version_id'],graph_sha256=execution['graph_sha256'],reused_atoms=1,new_atoms=0,
        original_history_sha256=imported['original_history_sha256'],limitations=LIMITATIONS,
        evidence_sha256={name:sha(directory/name) for name in names},reviewer_sha256=sha(__file__))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    report=review(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/quadratic_revision_publication_review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(eligible=True,proposed_tier=3,approved=False)))
