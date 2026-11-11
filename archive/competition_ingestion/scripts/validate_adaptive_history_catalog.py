"""Execute the exact catalog draft through the tested multi-output scenarios."""
import json
from pathlib import Path
from unittest.mock import patch

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from scripts.import_adaptive_history_draft import plan
from scripts.review_conditional_correction import ROOT, require, sha
import scripts.validate_adaptive_history_graph as execution


def validate():
    proposed=plan()
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
                         options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        document=db.execute('SELECT get_artifact_document(%s) AS d',(proposed['fqdn'],)).fetchone()['d']
        stored=db.execute('SELECT node_id,bound_artifact_fqdn,bound_version_content_hash,status FROM artifact_cdg_bindings '
                          'WHERE version_id=%s',(proposed['version_id'],)).fetchall()
        binding=proposed['bindings'][0]
        require(stored==[dict(node_id=binding['node_id'],bound_artifact_fqdn=binding['fqdn'],
            bound_version_content_hash=binding['content_hash'],status='active')],'Stored binding differs')
        ids=[proposed['artifact_id'],binding['artifact_id']]
        rows=db.execute('SELECT status,is_publishable FROM artifacts WHERE artifact_id=ANY(%s::uuid[])',(ids,)).fetchall()
        require(len(rows)==2 and all(r==dict(status='draft',is_publishable=False) for r in rows),'Draft state differs')
        require(not db.execute('SELECT 1 FROM catalog_artifacts_served WHERE artifact_id=ANY(%s::uuid[])',(ids,)).fetchone(),'Draft unexpectedly served')
    graph=_artifact_document_to_cdg(document,version_id=proposed['version_id'],content_hash=proposed['graph_sha256'],require_execution_envelope=True)
    with patch.object(execution,'build_adaptive_history_graph',return_value=graph):
        result=execution.validate()
    require(result['graph_sha256']==proposed['graph_sha256'],'Catalog execution identity differs')
    return dict(passed=True,approved=False,catalog_mutations=0,catalog_version_id=proposed['version_id'],
        graph_sha256=proposed['graph_sha256'],exact_binding_verified=True,draft_artifacts=2,served_artifacts=0,
        execution=result,validator_sha256=sha(Path(__file__)))


if __name__=='__main__':
    result=validate()
    (ROOT/'docs/reviews/adaptive_history_catalog_execution.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='execution'}))
