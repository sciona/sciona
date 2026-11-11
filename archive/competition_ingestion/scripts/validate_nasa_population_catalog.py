"""Execute catalog population graphs, including stored sparse inference in fresh children."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from sciona.services.catalog_artifact_retrieval import _artifact_document_to_cdg
from scripts.plan_nasa_population_drafts import ROOT,plan,sha
from scripts.check_nasa_population_catalog import check_staged
import scripts.validate_nasa_population_graphs as execution


def materialize(snapshot):
    return _artifact_document_to_cdg(snapshot['document'],version_id=snapshot['version_id'],
        content_hash=snapshot['graph_sha256'],require_execution_envelope=True)


def validate(source,approved=False):
    proposed=plan();snapshots={}
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        check_staged(db,proposed,approved=approved)
        new_ids=[item['artifact_id'] for item in proposed['atoms']+list(proposed['graphs'].values())]
        count=db.execute('SELECT count(DISTINCT artifact_id) AS n FROM catalog_artifacts_served '
            'WHERE artifact_id=ANY(%s::uuid[])',(new_ids,)).fetchone()['n']
        if count!=(len(new_ids) if approved else 0):raise ValueError('Population serving state differs')
        if approved:
            atom_ids=[item['artifact_id'] for item in proposed['atoms']+proposed['reused_atoms']]
            count=db.execute('SELECT count(*) AS n FROM catalog_atoms_served WHERE atom_id=ANY(%s::uuid[])',
                (atom_ids,)).fetchone()['n']
            if count!=len(atom_ids):raise ValueError('Each provider must be served once')
        for key,graph in proposed['graphs'].items():
            document=db.execute('SELECT get_artifact_document(%s) AS d',(graph['fqdn'],)).fetchone()['d']
            snapshots[key]=dict(document=document,version_id=graph['version_id'],graph_sha256=graph['graph_sha256'])
    graphs={key:materialize(snapshot) for key,snapshot in snapshots.items()}
    original_select=execution.select_population_graph
    def select(kind,payload):
        selected=original_select(kind,payload)
        return graphs[f"{kind}_{selected.metadata['active_population_count']}"]
    original_run=execution.subprocess.run;children=0
    def run(command,*args,**kwargs):
        nonlocal children
        if isinstance(command,list) and len(command)==4 and command[2]=='--child-directory' and Path(command[1]).name=='validate_nasa_population_graphs.py':
            directory=Path(command[3])
            count=json.loads((directory/'request.json').read_text())['count']
            (directory/'catalog_inference.json').write_text(json.dumps(snapshots[f'inference_{count}']))
            command=[command[0],str(Path(__file__).resolve()),'--child-directory',str(directory)]
            children+=1
        return original_run(command,*args,**kwargs)
    with patch.object(execution,'build_population_graph',side_effect=lambda kind,count:graphs[f'{kind}_{count}']), \
            patch.object(execution,'select_population_graph',side_effect=select), \
            patch.object(execution.subprocess,'run',side_effect=run):
        result=execution.validate(source)
    if not result['passed'] or children!=10 or result['predictions_compared']!=3520:
        raise ValueError('Full stored population graph replay required')
    expected={key:graph['graph_sha256'] for key,graph in proposed['graphs'].items()}
    executed={'training_10':result['training_graph_sha256'],
        **{f"inference_{case['active_populations']}":case['graph_sha256'] for case in result['inference_cases']}}
    if executed!=expected:raise ValueError('Executed catalog graph identities differ')
    return dict(passed=True,approved=approved,catalog_mutations=0,new_atoms=len(proposed['atoms']),
        reused_approved_atoms=len(proposed['reused_atoms']),cdgs=len(graphs),bindings=sum(len(g['bindings']) for g in proposed['graphs'].values()),
        catalog_inference_children=children,graph_sha256=expected,
        version_ids={key:graph['version_id'] for key,graph in proposed['graphs'].items()},
        execution=result,validator_sha256=sha(Path(__file__)),checker_sha256=sha(ROOT/'scripts/check_nasa_population_catalog.py'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-directory',type=Path)
    parser.add_argument('--child-directory',type=Path)
    parser.add_argument('--approved',action='store_true')
    args=parser.parse_args()
    if args.child_directory:
        snapshot=json.loads((args.child_directory/'catalog_inference.json').read_text())
        graph=materialize(snapshot)
        original_select=execution.select_population_graph
        def select(kind,payload):
            selected=original_select(kind,payload)
            if kind!='inference' or selected.metadata['active_population_count']!=graph.metadata['active_population_count']:
                raise ValueError('Stored child selection differs')
            return graph
        with patch.object(execution,'select_population_graph',side_effect=select):
            execution.child(args.child_directory)
    elif args.source_directory:
        report=validate(args.source_directory,args.approved)
        name='nasa_population_served_verification.json' if args.approved else 'nasa_population_catalog_execution.json'
        (ROOT/'docs/reviews'/name).write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({key:value for key,value in report.items() if key not in ('execution','version_ids','graph_sha256')}))
    else:parser.error('A source directory or child directory is required')
