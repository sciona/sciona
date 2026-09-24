"""Review the stored reconstructed momentum revision for scoped automated Tier 3 publication."""
import json
from pathlib import Path

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from scripts.check_momentum_revision import check
from scripts.stage_momentum_original_revision import ROOT,ARTIFACT,EXECUTION,sha,require
from scripts.audit_momentum_original_scope import audit
from scripts.validate_momentum_revision_database_gates import cases

LIMITATIONS=[
    'Automated Tier 3 reconstructed momentum realization under original identity; no Tier 1 human certification.',
    'Real finite vectors with identical nonempty shapes (...,3), in one orthonormal frame and consistent SI momentum units; no broadcasting.',
    'One existing approved provider is reused. Signed and zero components are supported; outputs are recoil vector and its squared Euclidean norm.',
    'Seven complete source equations and one explicitly reconstructed final dot-product expression retain the full five-step source scope. The missing symbolic RHS and incorrect vector LHS remain historical evidence.',
    'No literal source-AST or inference-rule parity claim. Momentum conservation is a caller premise; no energy, photon dispersion, scattering angle or full event validation.',
    'Exact rational converted-input differences are squared before independent float64 output rounding; squared norm need not equal a norm recomputed from the rounded recoil vector.',
    'Five stored runner cases cover 84 synthetic vectors, exactly matching independent 2500-digit Decimal arithmetic after float64 rounding.',
    'Exact zero and representable subnormals supported; nonzero values rounding to zero and nonfinite outputs rejected. No throughput claim.',
    'Provisioned in-process runtime and catalog metadata only; no HTTP, clean installation or redistribution qualification.',
    'Immutable source-version checks permit current publication state to change; original source rows and historical draft-only validator remain preserved.',
]


def review(source):
    directory=ROOT/'docs/reviews'
    scope=audit(source)
    require(scope==json.loads((directory/'physics_momentum_original_scope.json').read_text()),'Source scope changed')
    execution=json.loads((directory/'momentum_revision_catalog_execution.json').read_text())
    gates=json.loads((directory/'momentum_revision_database_gates.json').read_text())
    require(execution['passed'] and execution['catalog_mutations']==0 and execution['original_history_preserved'],'Stored execution required')
    numerical=execution['stored_execution']
    require(numerical['full_runner_cases']==5 and numerical['synthetic_vectors']==84
        and numerical['maximum_ulp_error']==0 and numerical['source_proof']==scope['source_proof'],'Numerical coverage differs')
    require(execution['validator_sha256']==sha(ROOT/'scripts/validate_momentum_revision_execution.py'),'Execution validator changed')
    for name,digest in numerical['implementation_sha256'].items():require(sha(ROOT/name)==digest,'Numerical implementation changed')
    for name,digest in numerical['source_proof']['implementation_sha256'].items():require(sha(ROOT/name)==digest,'Immutable source validator changed')
    require(execution['source_validator_sha256']==sha(ROOT/'scripts/validate_momentum_immutable_source.py'),'Source validator identity differs')
    require(gates['passed'] and gates['rollback_verified'] and gates['committed_catalog_mutations']==0
        and gates['rejected_faults']==[c[0] for c in cases(execution['version_id'])],'Negative gates incomplete')
    require(gates['validator_sha256']==sha(ROOT/'scripts/validate_momentum_revision_database_gates.py')
        and gates['checker_sha256']==sha(ROOT/'scripts/check_momentum_revision.py'),'Integrity code changed')
    require(gates['graph_sha256']==execution['graph_sha256'] and gates['version_id']==execution['version_id'],'Evidence identity differs')
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        state=db.execute('SELECT status,is_publishable FROM artifacts WHERE artifact_id=%s',(ARTIFACT,)).fetchone()
        graph,imported=check(db,state==dict(status='approved',is_publishable=True))
        require(imported['version_id']==execution['version_id'] and imported['graph_sha256']==numerical['graph_digest'],'Stored qualification differs')
        rows=db.execute("SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='momentum-corrected-community.v1' AND passed",(EXECUTION,)).fetchall()
        require(len(rows)==1,'Unique parent approval required')
        parent=rows[0]['details']
        require(parent['publication_tier']==3 and parent['review_source']=='automated','Parent tier differs')
        for name,digest in parent['evidence_sha256'].items():require(sha(directory/name)==digest,'Parent evidence changed')
    tests=json.loads((directory/'momentum_test_review.json').read_text())
    require(tests['test_results']==dict(tests=51,failures=0,errors=0,skipped=0),'Complete provider test qualification required')
    require(tests['provider_sha256']==numerical['provider_sha256'],'Provider test identity differs')
    for name,digest in tests['test_source_sha256'].items():require(sha(ROOT/name)==digest,'Provider test code changed')
    names=['physics_momentum_original_scope.json','momentum_original_revision_transaction.json',
        'momentum_original_revision_import.json','momentum_original_revision_repeat.json',
        'momentum_revision_catalog_execution.json','momentum_revision_database_gates.json','momentum_test_review.json','momentum_correction_review.md']
    return dict(eligible_for_approval_transaction=True,approved=False,proposed_tier=3,review_source='automated',catalog_mutations=0,
        artifact_id=ARTIFACT,version_id=execution['version_id'],graph_sha256=execution['graph_sha256'],reused_atoms=1,new_atoms=0,
        original_history_sha256=imported['original_history_sha256'],limitations=LIMITATIONS,
        evidence_sha256={name:sha(directory/name) for name in names},reviewer_sha256=sha(__file__))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    report=review(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/momentum_revision_publication_review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(eligible=True,proposed_tier=3,approved=False)))
