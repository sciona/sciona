"""Review the stored fixed Euler proof revision for scoped automated Tier 3 publication."""
import json
from pathlib import Path

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from scripts.check_euler_revision import check
from scripts.stage_euler_original_revision import ROOT,ARTIFACT,EXECUTION,sha,require
from scripts.audit_euler_original_scope import audit
from scripts.validate_euler_revision_database_gates import cases

LIMITATIONS=[
    'Automated Tier 3 fixed Euler proof realization under original identity; no Tier 1 human certification.',
    'Zero inputs and one JSON-safe certificate output, with the initial real-angle identity, four transformations and exact Equality(0,0).',
    'One approved provider is reused. This is a reusable fixed certificate component; no parameterized phasor computation, generic theorem proving or physical simulation claim.',
    'Five original parsed expressions reconcile with source equations. Four steps pass under explicit identity-specific interpretation of pi and the imaginary unit.',
    'The source imaginary-unit entry is labelled variable. This reviewed interpretation fixes it to I; literal source-AST parity remains false and historical failed evidence is preserved.',
    'The initial identity is checked through SymPy complex expansion, not derived from first principles.',
    'Two fresh stored-graph executions agree exactly with all interpreted source steps and produce identical JSON dictionary certificates.',
    'Provisioned in-process runtime and catalog metadata only; HTTP transport, clean installation and package redistribution remain unqualified.',
    'Current catalog state and version-bound semantic evidence govern selection; graph metadata preserves historical draft provenance.',
]


def review(source):
    directory=ROOT/'docs/reviews'
    scope=audit(source)
    require(scope==json.loads((directory/'physics_euler_original_scope.json').read_text()),'Source scope changed')
    execution=json.loads((directory/'euler_revision_catalog_execution.json').read_text())
    gates=json.loads((directory/'euler_revision_database_gates.json').read_text())
    require(execution['passed'] and execution['catalog_mutations']==0 and execution['original_history_preserved'],'Stored execution required')
    numerical=execution['stored_execution']
    require(numerical['full_runner_cases']==2 and numerical['proof_steps_per_run']==4
        and numerical['exact_source_interpreted_step_parity'] and numerical['source_interpretation']['graphs']==[scope['source_interpretation']],'Exact certificate coverage differs')
    require(execution['validator_sha256']==sha(ROOT/'scripts/validate_euler_revision_execution.py'),'Execution validator changed')
    for name,digest in numerical['implementation_sha256'].items():require(sha(ROOT/name)==digest,'Numerical implementation changed')
    require(gates['passed'] and gates['rollback_verified'] and gates['committed_catalog_mutations']==0
        and gates['rejected_faults']==[c[0] for c in cases(execution['version_id'])],'Negative gates incomplete')
    require(gates['validator_sha256']==sha(ROOT/'scripts/validate_euler_revision_database_gates.py')
        and gates['checker_sha256']==sha(ROOT/'scripts/check_euler_revision.py'),'Integrity code changed')
    require(gates['graph_sha256']==execution['graph_sha256'] and gates['version_id']==execution['version_id'],'Evidence identity differs')
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        state=db.execute('SELECT status,is_publishable FROM artifacts WHERE artifact_id=%s',(ARTIFACT,)).fetchone()
        graph,imported=check(db,state==dict(status='approved',is_publishable=True))
        require(imported['version_id']==execution['version_id'] and imported['graph_sha256']==numerical['graph_digest'],'Stored qualification differs')
        rows=db.execute("SELECT details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='euler-interpreted-community.v1' AND passed",(EXECUTION,)).fetchall()
        require(len(rows)==1,'Unique parent approval required')
        parent=rows[0]['details']
        require(parent['publication_tier']==3 and parent['review_source']=='automated','Parent tier differs')
        for name,digest in parent['evidence_sha256'].items():require(sha(directory/name)==digest,'Parent evidence changed')
    tests=json.loads((directory/'euler_test_review.json').read_text())
    require(tests['test_results']==dict(tests=10,failures=0,errors=0,skipped=0),'Complete provider test qualification required')
    require(tests['provider_sha256']==numerical['provider_sha256'],'Provider test identity differs')
    for name,digest in tests['test_source_sha256'].items():require(sha(ROOT/name)==digest,'Provider test code changed')
    names=['physics_euler_original_scope.json','euler_original_revision_transaction.json',
        'euler_original_revision_import.json','euler_original_revision_repeat.json',
        'euler_revision_catalog_execution.json','euler_revision_database_gates.json','euler_test_review.json','euler_execution_review.md']
    return dict(eligible_for_approval_transaction=True,approved=False,proposed_tier=3,review_source='automated',catalog_mutations=0,
        artifact_id=ARTIFACT,version_id=execution['version_id'],graph_sha256=execution['graph_sha256'],reused_atoms=1,new_atoms=0,
        original_history_sha256=imported['original_history_sha256'],limitations=LIMITATIONS,
        evidence_sha256={name:sha(directory/name) for name in names},reviewer_sha256=sha(__file__))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    report=review(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/euler_revision_publication_review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(eligible=True,proposed_tier=3,approved=False)))
