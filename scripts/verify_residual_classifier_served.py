"""Fresh served-catalog verification and execution after Tier 3 publication."""
import json
from pathlib import Path
from unittest.mock import patch

import psycopg
from psycopg.rows import dict_row
from dotenv import dotenv_values

from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from scripts.plan_residual_classifier_draft import plan
from scripts.review_conditional_correction import ROOT,require,sha
from scripts.validate_residual_classifier_database_gates import check_staged
import scripts.validate_residual_classifier_reuse as execution


def verify():
    proposed=plan()
    ids=[proposed['artifact_id']]+[atom['artifact_id'] for atom in proposed['atoms']]
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
                         options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        check_staged(db,proposed,approved=True)
        counts=db.execute('SELECT count(DISTINCT artifact_id) AS n FROM catalog_artifacts_served '
                          'WHERE artifact_id=ANY(%s::uuid[])',(ids,)).fetchone()
        require(counts['n']==17,'All artifacts must be served')
        require(db.execute('SELECT count(*) AS n FROM catalog_atoms_served WHERE atom_id=ANY(%s::uuid[])',
                (ids[1:],)).fetchone()['n']==16,'Each provider must be served once in legacy view')
        document=db.execute('SELECT get_artifact_document(%s) AS d',(proposed['fqdn'],)).fetchone()['d']
    graph=_artifact_document_to_cdg(document,version_id=proposed['version_id'],content_hash=proposed['graph_sha256'],
                                   require_execution_envelope=True)
    with patch.object(execution,'build_residual_classifier_graph',return_value=graph):
        result=execution.validate()
    require(result['graph_sha256']==proposed['graph_sha256'],'Executed served graph differs')
    return dict(passed=True,served_atoms=16,served_cdgs=1,trust_tier=3,source_intake_remains_draft=True,
        version_id=proposed['version_id'],graph_sha256=proposed['graph_sha256'],scenarios=result['scenarios'],
        validator_sha256=sha(Path(__file__)),catalog_mutations=0,
        limitations=result['limitations'][:2])


if __name__=='__main__':
    result=verify()
    (ROOT/'docs/reviews/residual_classifier_served_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
