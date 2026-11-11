"""Verify selectable Tier 3 lifecycle graphs and execute stored inference in fresh children."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from scripts.plan_nasa_lifecycle_drafts import ROOT,plan,sha
from scripts.validate_nasa_lifecycle_database_gates import check_staged
import scripts.validate_nasa_lifecycle_graphs as execution


def materialize(snapshot):
    return _artifact_document_to_cdg(snapshot['document'],version_id=snapshot['version_id'],
        content_hash=snapshot['graph_sha256'],require_execution_envelope=True)


def validate(source):
    proposed=plan()
    snapshots={}
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        check_staged(db,proposed,approved=True)
        atom_ids=[atom['artifact_id'] for atom in proposed['atoms']+proposed['reused_atoms']]
        ids=atom_ids+[graph['artifact_id'] for graph in proposed['graphs'].values()]
        count=db.execute('SELECT count(DISTINCT artifact_id) AS n FROM catalog_artifacts_served '
            'WHERE artifact_id=ANY(%s::uuid[])',(ids,)).fetchone()['n']
        if count!=len(ids):raise ValueError('All lifecycle artifacts must be selectable')
        count=db.execute('SELECT count(*) AS n FROM catalog_atoms_served WHERE atom_id=ANY(%s::uuid[])',
            (atom_ids,)).fetchone()['n']
        if count!=len(atom_ids):raise ValueError('All providers must be served once in the legacy view')
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
    # The shared execution validator predates publication. Its runtime checks
    # still apply, but this fresh catalog check establishes current approval.
    result['approved']=True
    result['limitations']=[item for item in result['limitations']
        if item!='New lifecycle atoms and graphs are not yet catalog-approved.']
    return dict(passed=True,approved=True,trust_tier=3,catalog_mutations=0,served_artifacts=len(ids),
        newly_approved_atoms=len(proposed['atoms']),served_cdgs=len(proposed['graphs']),source_intake_remains_draft=True,
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
        (ROOT/'docs/reviews/nasa_lifecycle_served_verification.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({key:value for key,value in report.items() if key!='execution'}))
    else:parser.error('A source directory or child directory is required')
