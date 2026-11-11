"""Execute stored lifecycle graphs, including the stored graph in fresh children."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from scripts.plan_nasa_lifecycle_drafts import ROOT,plan,check_parent,sha
import scripts.validate_nasa_lifecycle_graphs as execution


def materialize(snapshot):
    return _artifact_document_to_cdg(snapshot['document'],version_id=snapshot['version_id'],
        content_hash=snapshot['graph_sha256'],require_execution_envelope=True)


def validate(source):
    proposed=plan()
    snapshots={}
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        check_parent(db,proposed['parent_domain'])
        ids=[atom['artifact_id'] for atom in proposed['atoms']]+[graph['artifact_id'] for graph in proposed['graphs'].values()]
        rows=db.execute('SELECT status,is_publishable FROM artifacts WHERE artifact_id=ANY(%s::uuid[])',(ids,)).fetchall()
        if len(rows)!=len(ids) or any(row!=dict(status='draft',is_publishable=False) for row in rows):
            raise ValueError('Expected lifecycle drafts')
        if db.execute('SELECT 1 FROM catalog_artifacts_served WHERE artifact_id=ANY(%s::uuid[])',(ids,)).fetchone():
            raise ValueError('Lifecycle draft unexpectedly served')
        for atom in proposed['atoms']:
            row=db.execute('SELECT artifact_id,content_hash,trust_tier FROM artifact_versions WHERE version_id=%s',(atom['version_id'],)).fetchone()
            if not row or str(row['artifact_id'])!=atom['artifact_id'] or row['content_hash']!=atom['content_hash'] or row['trust_tier']!=3:
                raise ValueError('Lifecycle provider version differs')
        for kind,graph in proposed['graphs'].items():
            bindings=db.execute('SELECT node_id,bound_artifact_fqdn,bound_version_content_hash,status FROM artifact_cdg_bindings '
                'WHERE version_id=%s ORDER BY node_id',(graph['version_id'],)).fetchall()
            expected=sorted([dict(node_id=b['node_id'],bound_artifact_fqdn=b['fqdn'],
                bound_version_content_hash=b['content_hash'],status='active') for b in graph['bindings']],key=lambda b:b['node_id'])
            if bindings!=expected:raise ValueError('Lifecycle catalog bindings differ')
            document=db.execute('SELECT get_artifact_document(%s) AS d',(graph['fqdn'],)).fetchone()['d']
            snapshots[kind]=dict(document=document,version_id=graph['version_id'],graph_sha256=graph['graph_sha256'])
    graphs={kind:materialize(snapshot) for kind,snapshot in snapshots.items()}
    original_run=execution.subprocess.run
    child_count=0
    def run(command,*args,**kwargs):
        nonlocal child_count
        if isinstance(command,list) and len(command)==4 and command[2]=='--child-directory' and Path(command[1]).name=='validate_nasa_lifecycle_graphs.py':
            directory=Path(command[3])
            (directory/'catalog_inference.json').write_text(json.dumps(snapshots['inference']))
            command=[command[0],str(Path(__file__).resolve()),'--child-directory',str(directory)]
            child_count+=1
        return original_run(command,*args,**kwargs)
    with patch.object(execution,'build_nasa_training_graph',return_value=graphs['training']), \
            patch.object(execution,'build_nasa_inference_graph',return_value=graphs['inference']), \
            patch.object(execution.subprocess,'run',side_effect=run):
        result=execution.validate(source)
    if child_count!=10 or not result['passed'] or result['source_predictions']!=640:
        raise ValueError('Complete fresh catalog lifecycle replay required')
    return dict(passed=True,approved=False,catalog_mutations=0,draft_artifacts=len(ids),
        reused_approved_atoms=len(proposed['reused_atoms']),catalog_inference_children=child_count,
        version_ids={kind:graph['version_id'] for kind,graph in proposed['graphs'].items()},
        graph_sha256={kind:graph['graph_sha256'] for kind,graph in proposed['graphs'].items()},
        execution=result,validator_sha256=sha(Path(__file__)))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-directory',type=Path)
    parser.add_argument('--child-directory',type=Path)
    args=parser.parse_args()
    if args.child_directory:
        snapshot=json.loads((args.child_directory/'catalog_inference.json').read_text())
        with patch.object(execution,'build_nasa_inference_graph',return_value=materialize(snapshot)):
            execution.child(args.child_directory)
    elif args.source_directory:
        report=validate(args.source_directory)
        (ROOT/'docs/reviews/nasa_lifecycle_catalog_execution.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({key:value for key,value in report.items() if key!='execution'}))
    else:parser.error('A source directory or child directory is required')
