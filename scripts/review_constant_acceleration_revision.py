"""Review the stored corrected constant_acceleration revision for scoped automated Tier 3 publication."""
import json
from pathlib import Path

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from scripts.check_constant_acceleration_revision import check
from scripts.stage_constant_acceleration_original_revision import ROOT,ARTIFACT,EXECUTION,sha,require
from scripts.audit_constant_acceleration_original_scope import audit
from scripts.validate_constant_acceleration_revision_database_gates import cases

LIMITATIONS=[
    'Automated Tier 3 corrected constant-acceleration realization under original identity; no Tier 1 human certification.',
    'Finite signed one-dimensional SI initial velocity and constant acceleration with nonnegative elapsed time. Identical nonempty array shapes, including scalars; no broadcasting.',
    'One existing approved provider is reused. Signed final velocity, displacement and average velocity use exact rational arithmetic on converted float64 inputs before independent output rounding.',
    'All 23 reconstructed source steps and 24 expressions are retained as proof provenance. One missing average-velocity equation and a damaged squared-velocity AST are recovered from pinned public bytes.',
    'Original time quotients require nonzero time; acceleration-dividing branches require nonzero acceleration. Independent polynomial integration establishes the zero-time and zero-acceleration extensions.',
    'At zero time average velocity is defined by continuity. During reversal, displacement is not path length. Squared velocity does not uniquely determine signed velocity.',
    'Six stored runner cases cover 29 synthetic states with exact agreement against an independent 2500-digit Decimal reference for all three outputs.',
    'Exact zeros and representable subnormals supported; nonzero underflow and overflow rejected. No variable acceleration, collision detection, empirical model-accuracy or throughput claim.',
    'Provisioned in-process runtime and catalog metadata only; no HTTP, clean-install or redistribution qualification.',
    'Immutable source-version checks are independent of publication status; original source rows and historical validator remain preserved.',
]


def review(source):
    directory=ROOT/'docs/reviews'
    scope=audit(source)
    require(scope==json.loads((directory/'physics_constant_acceleration_original_scope.json').read_text()),'Source scope changed')
    execution=json.loads((directory/'constant_acceleration_revision_catalog_execution.json').read_text())
    gates=json.loads((directory/'constant_acceleration_revision_database_gates.json').read_text())
    require(execution['passed'] and execution['catalog_mutations']==0 and execution['original_history_preserved'],'Stored execution required')
    numerical=execution['stored_execution']
    require(numerical['full_runner_cases']==6 and numerical['synthetic_states']==29
        and numerical['maximum_ulp_error']==0 and numerical['source_proof']==scope['source_proof'],'Numerical coverage differs')
    require(execution['validator_sha256']==sha(ROOT/'scripts/validate_constant_acceleration_revision_execution.py'),'Execution validator changed')
    for name,digest in numerical['implementation_sha256'].items():require(sha(ROOT/name)==digest,'Numerical implementation changed')
    for name,digest in numerical['source_proof']['implementation_sha256'].items():require(sha(ROOT/name)==digest,'Immutable source validator changed')
    require(execution['source_validator_sha256']==sha(ROOT/'scripts/validate_constant_acceleration_immutable_source.py'),'Source validator identity differs')
    require(gates['passed'] and gates['rollback_verified'] and gates['committed_catalog_mutations']==0
        and gates['rejected_faults']==[c[0] for c in cases(execution['version_id'])],'Negative gates incomplete')
    require(gates['validator_sha256']==sha(ROOT/'scripts/validate_constant_acceleration_revision_database_gates.py')
        and gates['checker_sha256']==sha(ROOT/'scripts/check_constant_acceleration_revision.py'),'Integrity code changed')
    require(gates['graph_sha256']==execution['graph_sha256'] and gates['version_id']==execution['version_id'],'Evidence identity differs')
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        state=db.execute('SELECT status,is_publishable FROM artifacts WHERE artifact_id=%s',(ARTIFACT,)).fetchone()
        graph,imported=check(db,state==dict(status='approved',is_publishable=True))
        require(imported['version_id']==execution['version_id'] and imported['graph_sha256']==numerical['graph_digest'],'Stored qualification differs')
        rows=db.execute("SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='constant_acceleration-corrected-community.v1' AND passed",(EXECUTION,)).fetchall()
        require(len(rows)==1,'Unique parent approval required')
        parent=rows[0]['details']
        require(parent['publication_tier']==3 and parent['review_source']=='automated','Parent tier differs')
        for name,digest in parent['evidence_sha256'].items():require(sha(directory/name)==digest,'Parent evidence changed')
    tests=json.loads((directory/'constant_acceleration_test_review.json').read_text())
    require(tests['test_results']==dict(tests=49,failures=0,errors=0,skipped=0),'Complete provider test qualification required')
    require(tests['provider_sha256']==numerical['provider_sha256'],'Provider test identity differs')
    for name,digest in tests['test_source_sha256'].items():require(sha(ROOT/name)==digest,'Provider test code changed')
    names=['physics_constant_acceleration_original_scope.json','constant_acceleration_original_revision_transaction.json',
        'constant_acceleration_original_revision_import.json','constant_acceleration_original_revision_repeat.json',
        'constant_acceleration_revision_catalog_execution.json','constant_acceleration_revision_database_gates.json','constant_acceleration_test_review.json','constant_acceleration_correction_review.md']
    return dict(eligible_for_approval_transaction=True,approved=False,proposed_tier=3,review_source='automated',catalog_mutations=0,
        artifact_id=ARTIFACT,version_id=execution['version_id'],graph_sha256=execution['graph_sha256'],reused_atoms=1,new_atoms=0,
        original_history_sha256=imported['original_history_sha256'],limitations=LIMITATIONS,
        evidence_sha256={name:sha(directory/name) for name in names},reviewer_sha256=sha(__file__))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    report=review(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/constant_acceleration_revision_publication_review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(eligible=True,proposed_tier=3,approved=False)))
