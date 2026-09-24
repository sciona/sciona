"""Execute the exact domain draft retrieved from the catalog with shared atoms."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from scripts.plan_nasa_domain_draft import ROOT, plan, check_reused, sha
import scripts.validate_nasa_domain_graph as execution


def validate(source):
    proposed = plan()
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        check_reused(db,proposed['numerical_core'])
        document=db.execute('SELECT get_artifact_document(%s) AS d',(proposed['fqdn'],)).fetchone()['d']
        stored=db.execute('SELECT node_id,bound_artifact_fqdn,bound_version_content_hash,status FROM artifact_cdg_bindings '
                          'WHERE version_id=%s ORDER BY node_id',(proposed['version_id'],)).fetchall()
        expected=sorted([dict(node_id=b['node_id'],bound_artifact_fqdn=b['fqdn'],
            bound_version_content_hash=b['content_hash'],status='active') for b in proposed['bindings']],key=lambda b:b['node_id'])
        if stored!=expected:raise ValueError('Stored domain bindings differ')
        ids=[proposed['artifact_id']]+[atom['artifact_id'] for atom in proposed['atoms']]
        rows=db.execute('SELECT status,is_publishable FROM artifacts WHERE artifact_id=ANY(%s::uuid[])',(ids,)).fetchall()
        if len(rows)!=len(ids) or any(row!=dict(status='draft',is_publishable=False) for row in rows):
            raise ValueError('Expected non-publishable domain drafts')
        if db.execute('SELECT 1 FROM catalog_artifacts_served WHERE artifact_id=ANY(%s::uuid[])',(ids,)).fetchone():
            raise ValueError('Domain draft unexpectedly served')
        for atom in proposed['atoms']:
            row=db.execute('SELECT artifact_id,content_hash,trust_tier FROM artifact_versions WHERE version_id=%s',
                           (atom['version_id'],)).fetchone()
            if row is None or str(row['artifact_id'])!=atom['artifact_id'] or row['content_hash']!=atom['content_hash'] or row['trust_tier']!=3:
                raise ValueError('Domain provider version differs')
        core=proposed['numerical_core']
        dependency=db.execute('SELECT optional,binding_metadata FROM artifact_dependencies WHERE dependent_version_id=%s '
            'AND dependency_artifact_fqdn=%s AND dependency_content_hash=%s',
            (proposed['version_id'],core['fqdn'],core['graph_sha256'])).fetchone()
        if not dependency or dependency['optional'] or dependency['binding_metadata']['source_version_id']!=core['version_id']:
            raise ValueError('Mandatory numerical graph provenance differs')
    graph=_artifact_document_to_cdg(document,version_id=proposed['version_id'],content_hash=proposed['graph_sha256'],
                                   require_execution_envelope=True)
    with patch.object(execution,'build_nasa_domain_graph',return_value=graph):
        result=execution.validate(source)
    if result['graph_sha256']!=proposed['graph_sha256']:
        raise ValueError('Executed stored domain graph differs')
    return dict(passed=True,approved=False,catalog_mutations=0,version_id=proposed['version_id'],
        graph_sha256=proposed['graph_sha256'],draft_artifacts=len(ids),reused_approved_atoms=len(proposed['reused_atoms']),
        exact_bindings_verified=len(stored),execution=result,validator_sha256=sha(Path(__file__)))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-directory',type=Path,required=True)
    args=parser.parse_args()
    report=validate(args.source_directory)
    (ROOT/'docs/reviews/nasa_domain_catalog_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key!='execution'}))
