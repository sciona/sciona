"""Execute the staged original-identity two-body graph using the existing precision oracle."""
import argparse
import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from scripts.stage_two_body_original_revision import ROOT,ARTIFACT,ORIGINAL,EXECUTION,history,sha,require
from scripts.audit_two_body_original_scope import audit
from sciona.physics_ingest.two_body_execution import build_two_body_execution
from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from sciona.services.execution_graph_codec import encode_execution_graph
import scripts.validate_two_body_execution as numerical


def stored_graph(imported):
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        require(history(db)==imported['original_history_sha256'],'Historical source records changed')
        state=db.execute('SELECT fqdn,status,is_publishable FROM artifacts WHERE artifact_id=%s',(ARTIFACT,)).fetchone()
        require(state and state['status']=='draft' and not state['is_publishable'],'Expected staged source artifact')
        latest=db.execute('SELECT version_id::text FROM artifact_versions WHERE artifact_id=%s AND is_latest',(ARTIFACT,)).fetchall()
        require(latest==[dict(version_id=ORIGINAL)],'Staging changed latest version')
        version=db.execute('SELECT content_hash,trust_tier,is_latest FROM artifact_versions WHERE version_id=%s AND artifact_id=%s',(imported['version_id'],ARTIFACT)).fetchone()
        require(version==dict(content_hash=imported['graph_sha256'],trust_tier=3,is_latest=False),'Candidate version differs')
        document=db.execute('SELECT get_artifact_document(%s) AS d',(state['fqdn'],)).fetchone()['d']
        graph=_artifact_document_to_cdg(document,version_id=imported['version_id'],content_hash=imported['graph_sha256'],require_execution_envelope=True)
    expected=build_two_body_execution().model_copy(deep=True)
    expected.metadata.update(original_artifact_id=ARTIFACT,original_version_id=ORIGINAL,qualified_numerical_parent_version_id=EXECUTION,
        revision_scope='Corrected complete circular Newtonian two-body period under original identity; original and repaired source graphs remain immutable provenance; no literal source-proof parity claim.')
    require(expected==graph and encode_execution_graph(graph)[0]==imported['graph_sha256'],'Stored graph differs from qualified numerical body')
    return graph


async def validate(source):
    imported=json.loads((ROOT/'docs/reviews/two_body_original_revision_import.json').read_text())
    require(imported['stager_sha256']==sha(ROOT/'scripts/stage_two_body_original_revision.py'),'Stager code changed')
    scope=audit(source)
    require(scope==json.loads((ROOT/'docs/reviews/physics_two_body_original_scope.json').read_text())
        and imported['scope_audit_sha256']==sha(ROOT/'docs/reviews/physics_two_body_original_scope.json'),'Source qualification changed')
    graph=stored_graph(imported)
    # Substitute the explicitly selected stored graph at the numerical runner's
    # construction boundary. Source proof checks and the independent mpmath oracle run unchanged.
    with patch.object(numerical,'build_two_body_execution',return_value=graph):
        execution=await numerical.validate(ROOT,source/'symbols.cypher',source/'infrules.cypher')
    require(execution['graph_digest']==imported['graph_sha256'] and execution['full_runner_cases']==5
        and execution['synthetic_numeric_cases']==208 and execution['maximum_ulp_error']<=1,'Stored precision qualification differs')
    require(stored_graph(imported)==graph,'Stored graph changed during execution')
    return dict(passed=True,approved=False,catalog_mutations=0,version_id=imported['version_id'],graph_sha256=imported['graph_sha256'],
        original_history_preserved=True,stored_execution=execution,validator_sha256=sha(__file__),
        scope_audit_sha256=imported['scope_audit_sha256'],limitations=[
            'This validates stored graph execution and original history. Complete provider/port/provenance and corruption gates remain separate publication prerequisites.',
            'No original artifact approval or literal source-rule parity; caller establishes circular Newtonian model applicability and SI units.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    report=asyncio.run(validate(parser.parse_args().source_directory))
    (ROOT/'docs/reviews/two_body_revision_catalog_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,approved=False,stored_numeric_cases=208,maximum_ulp_error=report['stored_execution']['maximum_ulp_error'])))
