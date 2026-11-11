"""Verify publication rollback after metadata writes and full activation."""
import hashlib
import json

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from scripts import promote_stacking_complete as publisher
from scripts.check_stacking_complete_catalog import ROOT, qualification, check_candidate, require, sha
from scripts.validate_stacking_complete_transaction import snapshot as draft_snapshot


def snapshot(scope,specs):
    result=draft_snapshot()
    owners=[target[0] for target in publisher.targets(scope)]
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
                         options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        check_candidate(db,scope,specs)
        rows=db.execute('SELECT to_jsonb(t) AS row FROM artifact_descriptions t WHERE artifact_id=ANY(%s::uuid[]) '
                        'ORDER BY to_jsonb(t)::text',(owners,)).fetchall()
        result['artifact_descriptions']=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
        require(not db.execute('SELECT 1 FROM artifact_audit_evidence WHERE artifact_id=ANY(%s::uuid[]) AND runner_version=%s',
                               (owners,publisher.RUNNER)).fetchone(),'Publication evidence survived rollback')
    return result


def main():
    scope,specs=qualification(); before=snapshot(scope,specs)
    dry=publisher.promote()
    require(not dry['already_approved'] and dry['rows_created']==139 and dry['rows_updated']==58,
            'Unpublished full transaction baseline required')
    require(snapshot(scope,specs)==before,'Publication rollback changed catalog')
    for phase in ('metadata','activation'):
        try:
            publisher.promote(fail_at=phase)
        except RuntimeError as error:
            require(str(error)=='injected publication '+phase+' failure','Unexpected publication failure')
        else:
            raise ValueError('Publication failure did not fire: '+phase)
        require(snapshot(scope,specs)==before,'Publication failure rollback differs: '+phase)
    report=dict(passed=True,rollback_verified=True,metadata_failure_rolled_back=True,activation_failure_rolled_back=True,
                committed_catalog_mutations=0,snapshot_tables=len(before),dry_run=dry,
                publisher_sha256=sha(publisher.__file__),validator_sha256=sha(__file__),
                review_sha256=sha(ROOT/'docs/reviews/competition_otto_decomposed_publication_review.json'),
                limitations=['Rollback-only publication transaction; persisted approval and served execution remain separate.'])
    (ROOT/'docs/reviews/competition_otto_decomposed_publication_transaction.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,activation_failure_rolled_back=True,committed_catalog_mutations=0)))


if __name__=='__main__':
    main()
