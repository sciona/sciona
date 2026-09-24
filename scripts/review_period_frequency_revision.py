"""Scoped automated Tier 3 review of the original period/frequency revision."""
import json
from pathlib import Path
import importlib.metadata

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from scripts.check_period_frequency_revision import check
from scripts.stage_period_frequency_original_revision import ROOT,ARTIFACT,EXECUTION,sha,require
from scripts.audit_period_frequency_original_scope import audit
from scripts.validate_period_frequency_revision_database_gates import cases
from sciona.physics_ingest.pdg_evidence import _digest

LIMITATIONS=[
    'Automated Tier 3 terminal numerical realization under the original source artifact identity; no Tier 1 human certification.',
    'Convert supplied finite positive period in seconds to ordinary frequency in hertz. Scalar and array shape are preserved, including empty arrays. Invalid types, nonpositive/nonfinite inputs and floating-point overflow/underflow are rejected.',
    'The original source equations and corrected source equations are rationally equivalent. Both algebraic steps and nonzero conditions were freshly replayed; the numerical graph evaluates the terminal relation, while historical proof versions remain immutable.',
    'One existing approved provider is reused across periodic motion, oscillating circuits and recurring signals with explicit seconds/hertz contracts. No domain-specific dataset, new atom or fitted model is introduced.',
    'This operation does not estimate periods from signals or compute angular frequency. Callers supply an applicable cycle duration.',
    '256 independent synthetic measurement cases, eight additional synthetic values and nine invalid-domain cases qualify the stored revision; this does not establish broad empirical effectiveness.',
    'Existing provisioned in-process runtime and catalog metadata only. HTTP transport, clean installation, package/binary redistribution and Tier 2 empirical claims remain unqualified.',
    'Graph metadata retains its immutable draft provenance. Current catalog status and version-bound semantic approval govern selection.',
]


def review(source):
    directory=ROOT/'docs/reviews'
    scope=audit(source)
    require(scope==json.loads((directory/'physics_period_frequency_original_scope.json').read_text()),'Fresh source scope differs')
    execution=json.loads((directory/'period_frequency_revision_catalog_execution.json').read_text())
    gates=json.loads((directory/'period_frequency_revision_database_gates.json').read_text())
    require(execution['passed'] and execution['catalog_mutations']==0 and execution['independent_measurement_cases']==256
        and execution['additional_synthetic_values']==8 and len(execution['invalid_cases_rejected'])==9,'Stored numerical qualification required')
    require(all(c['passed'] and c['input_unmodified'] for c in execution['valid_cases']),'Numerical boundary validation failed')
    require(gates['passed'] and gates['rollback_verified'] and gates['committed_catalog_mutations']==0
        and gates['rejected_faults']==[c[0] for c in cases(execution['version_id'])],'Database negative qualification required')
    for report,filename in [(execution,'validate_period_frequency_revision_execution.py'),(gates,'validate_period_frequency_revision_database_gates.py')]:
        require(report['validator_sha256']==sha(ROOT/'scripts'/filename) and report['checker_sha256']==sha(ROOT/'scripts/check_period_frequency_revision.py'),'Qualification code changed')
    require(gates['version_id']==execution['version_id'] and gates['graph_sha256']==execution['graph_sha256'],'Qualification version differs')
    for filename,digest in execution['source_sha256'].items():require(sha(ROOT/filename)==digest,'Execution source changed')
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        state=db.execute('SELECT status,is_publishable FROM artifacts WHERE artifact_id=%s',(ARTIFACT,)).fetchone()
        graph,imported=check(db,state==dict(status='approved',is_publishable=True))
        require(imported['version_id']==execution['version_id'] and imported['graph_sha256']==execution['graph_sha256'],'Selected qualification differs')
        parents=db.execute("SELECT evidence_id,details FROM artifact_audit_evidence WHERE version_id=%s AND runner_version='period-frequency-community-approval.v1' AND passed",(EXECUTION,)).fetchall()
        require(len(parents)==1,'Unique parent approval required')
        parent=parents[0];details=parent['details']
        require(details['publication_tier']==3 and details['review_source']=='automated','Parent tier differs')
        for identity,hash_key in [('source_replay_evidence_id','source_replay_sha256'),('execution_evidence_id','execution_evidence_sha256'),('provider_approval_evidence_id','provider_approval_sha256')]:
            row=db.execute('SELECT passed,details FROM artifact_audit_evidence WHERE evidence_id=%s',(details[identity],)).fetchone()
            require(row and row['passed'] and _digest(row['details'])==details[hash_key],'Parent approval chain differs')
    names=['physics_period_frequency_original_scope.json','period_frequency_original_revision_transaction.json',
        'period_frequency_original_revision_import.json','period_frequency_original_revision_repeat.json',
        'period_frequency_revision_catalog_execution.json','period_frequency_revision_database_gates.json']
    return dict(eligible_for_approval_transaction=True,approved=False,proposed_tier=3,review_source='automated',
        artifact_id=ARTIFACT,version_id=execution['version_id'],graph_sha256=execution['graph_sha256'],
        original_history_sha256=imported['original_history_sha256'],reused_atoms=1,new_atoms=0,limitations=LIMITATIONS,
        parent_approval_evidence_id=str(parent['evidence_id']),parent_approval_sha256=_digest(details),
        dependency_versions={name:importlib.metadata.version(name) for name in ['numpy','sympy','psycopg']},
        dependency_scope='Versions observed in the exercised provisioned runtime, not a clean-install or redistribution dependency closure.',
        evidence_sha256={name:sha(directory/name) for name in names},reviewer_sha256=sha(__file__),catalog_mutations=0)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    report=review(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/period_frequency_revision_publication_review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(eligible=True,proposed_tier=3,approved=False,reused_atoms=1)))
