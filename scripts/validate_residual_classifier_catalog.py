"""Execute the stored numerical draft and verify every exact provider binding."""
import json
from pathlib import Path
from unittest.mock import patch

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from scripts.plan_residual_classifier_draft import ROOT, plan, sha
import scripts.validate_residual_classifier_reuse as execution


def validate():
    proposed = plan()
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'], row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        document = db.execute('SELECT get_artifact_document(%s) AS d', (proposed['fqdn'],)).fetchone()['d']
        stored = db.execute('SELECT node_id,bound_artifact_fqdn,bound_version_content_hash,status FROM artifact_cdg_bindings '
                            'WHERE version_id=%s ORDER BY node_id', (proposed['version_id'],)).fetchall()
        expected = sorted([dict(node_id=b['node_id'], bound_artifact_fqdn=b['fqdn'],
            bound_version_content_hash=b['content_hash'], status='active') for b in proposed['bindings']], key=lambda b:b['node_id'])
        if stored != expected:
            raise ValueError('Stored node bindings differ')
        ids = [proposed['artifact_id']] + [atom['artifact_id'] for atom in proposed['atoms']]
        rows = db.execute('SELECT status,is_publishable FROM artifacts WHERE artifact_id=ANY(%s::uuid[])', (ids,)).fetchall()
        if len(rows) != len(ids) or any(row != dict(status='draft', is_publishable=False) for row in rows):
            raise ValueError('Expected non-publishable drafts')
        if db.execute('SELECT 1 FROM catalog_artifacts_served WHERE artifact_id=ANY(%s::uuid[])', (ids,)).fetchone():
            raise ValueError('Draft unexpectedly served')
        for atom in proposed['atoms']:
            version = db.execute('SELECT artifact_id,content_hash,trust_tier FROM artifact_versions WHERE version_id=%s',
                                 (atom['version_id'],)).fetchone()
            if version is None or str(version['artifact_id']) != atom['artifact_id'] or version['content_hash'] != atom['content_hash'] or version['trust_tier'] != 3:
                raise ValueError('Stored provider version differs')
    graph = _artifact_document_to_cdg(document, version_id=proposed['version_id'],
        content_hash=proposed['graph_sha256'], require_execution_envelope=True)
    with patch.object(execution, 'build_residual_classifier_graph', return_value=graph):
        result = execution.validate()
    if result['graph_sha256'] != proposed['graph_sha256']:
        raise ValueError('Executed catalog graph differs')
    return dict(passed=True, approved=False, catalog_mutations=0, catalog_version_id=proposed['version_id'],
        graph_sha256=proposed['graph_sha256'], exact_bindings_verified=len(stored),
        draft_artifacts=len(ids), served_artifacts=0, execution=result, validator_sha256=sha(Path(__file__)))


if __name__ == '__main__':
    report = validate()
    (ROOT/'docs/reviews/residual_classifier_catalog_execution.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key!='execution'}))
