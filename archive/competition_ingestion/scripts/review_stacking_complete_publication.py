"""Tier 3 publication readiness for the corrected original Otto identity."""
import json

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from scripts.check_stacking_complete_catalog import ROOT, qualification, check_candidate, require, sha
from scripts.promote_otto_execution import check_hashes
from scripts.validate_stacking_complete_catalog_gates import fault_cases


def review():
    scope,specs=qualification(); directory=ROOT/'docs/reviews'
    names=['competition_otto_decomposed_review_gates.json','competition_otto_decomposed_draft_transaction.json',
           'competition_otto_decomposed_catalog_gates.json']
    offline,transaction,catalog=[json.loads((directory/name).read_text()) for name in names]
    require(offline['status']=='offline_review_gates_passed' and len(offline['rejected_scenarios'])==21
            and offline['pre_catalog_baseline_passed_before_and_after'], 'Offline review gates differ')
    check_hashes(ROOT,offline['sha256'])
    require(transaction['passed'] and transaction['rollback_verified'] and transaction['committed_catalog_mutations']==0
            and transaction['injected_failures_rolled_back']==[1,431,862], 'Staging rollback evidence differs')
    require(transaction['stager_sha256']==sha(ROOT/'scripts/stage_stacking_complete.py')
            and transaction['validator_sha256']==sha(ROOT/'scripts/validate_stacking_complete_transaction.py'),
            'Staging qualification code changed')
    require(catalog['passed'] and catalog['rollback_verified'] and catalog['committed_catalog_mutations']==0
            and catalog['rejected_faults']==[case[0] for case in fault_cases(scope)], 'Catalog fault coverage differs')
    require(catalog['checker_sha256']==sha(ROOT/'scripts/check_stacking_complete_catalog.py')
            and catalog['validator_sha256']==sha(ROOT/'scripts/validate_stacking_complete_catalog_gates.py'),
            'Catalog qualification code changed')
    for report in (transaction,catalog):
        require(report['scope_sha256']==sha(directory/'competition_otto_decomposed_complete_scope.json'), 'Scope evidence changed')
    require(catalog['version_id']==scope['corrected_version_id'] and catalog['graph_sha256']==scope['graph_sha256']
            and catalog['nodes']==62 and catalog['providers']==27, 'Catalog qualification identity differs')
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
                         options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        state=db.execute('SELECT status FROM artifacts WHERE artifact_id=%s',(scope['artifact_id'],)).fetchone()
        check_candidate(db,scope,specs,approved=state['status']=='approved')
    return dict(status='ready_for_tier_3_publication_transaction',tier=3,review_source='automated',
        artifact_id=scope['artifact_id'],version_id=scope['corrected_version_id'],graph_sha256=scope['graph_sha256'],
        providers=27,nodes=62,edges=198,source_history_preserved=True,synthetic_only=True,
        reuse_scope=scope['reuse_scope'],limitations=scope['limitations']+[
            'Publication qualifies the corrected executable version, not the original six-stage conceptual graph.',
            'Native execution is qualified by the complete 9,900-fit run and exact catalog graph/provider identity; staging and fault checks do not repeat training.',
            'Publication rollback/idempotence and fresh served execution require their own evidence.'],
        scope_sha256=sha(directory/'competition_otto_decomposed_complete_scope.json'),
        evidence_sha256={name:sha(directory/name) for name in names},reviewer_sha256=sha(__file__))


if __name__=='__main__':
    result=review()
    (ROOT/'docs/reviews/competition_otto_decomposed_publication_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],tier=3,providers=27,nodes=62)))
