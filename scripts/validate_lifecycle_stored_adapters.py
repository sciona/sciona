"""Re-execute qualified workflow sections using stored lifecycle provider ports.

Topology remains the qualified local topology; this is not served-CDG retrieval.
Training wiring uses its explicit recording backend, while inference restores
the previously qualified native models. All domain adapter code runs normally.
"""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from sciona.architect.models import IOSpec
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
import sciona.visualizer.runner as runner
from scripts.plan_nasa_first_lifecycle_providers import ROOT,plan,sha
from scripts.validate_nasa_first_lifecycle_database_gates import check_staged
from scripts.validate_nasa_first_raw_training_graph import validate as raw_validate
from scripts.validate_nasa_first_population_training_wiring import validate as wiring_validate
from scripts.validate_nasa_first_policy_state_graphs import validate as state_validate


def validate(source,checkpoint):
    proposed=plan();stored={}
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
                        options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        check_staged(db,proposed)
        for atom in proposed['atoms']:
            rows=db.execute('SELECT direction,name,type_desc,constraints,required,default_value_repr,dim_signature FROM artifact_io_specs WHERE version_id=%s ORDER BY direction,ordinal',
                            (atom['version_id'],)).fetchall()
            stored[atom['runtime_fqdn']]=dict(atom=atom,ports={side:[IOSpec(**{k:v for k,v in row.items() if k!='direction'})
                for row in rows if row['direction']==direction] for side,direction in [('inputs','input'),('outputs','output')]})
    original=runner.CDGExecutionSession.execute
    traces=[];seen=set()

    async def execute(session,*args,**kwargs):
        graph=kwargs['cdg'].model_copy(deep=True)
        bindings={}
        for node in graph.nodes:
            item=stored.get(node.matched_primitive)
            if item is None:continue
            for side in ['inputs','outputs']:
                local=getattr(node,side);canonical=item['ports'][side]
                if [p.name for p in local]!=[p.name for p in canonical]:
                    raise ValueError('Stored adapter port order differs')
                setattr(node,side,[p.model_copy(deep=True) for p in canonical])
            bindings[node.node_id]={key:item['atom'][key] for key in ['version_id','content_hash','runtime_fqdn']}
        nodes={node.node_id:node for node in graph.nodes}
        for edge in graph.edges:
            edge.source_type=next(p.type_desc for p in nodes[edge.source_id].outputs if p.name==edge.output_name)
            edge.target_type=next(p.type_desc for p in nodes[edge.target_id].inputs if p.name==edge.input_name)
        graph.metadata['stored_lifecycle_bindings']=bindings
        digest,encoded_nodes,encoded_edges=encode_execution_graph(graph)
        restored=decode_execution_graph(encoded_nodes,encoded_edges,digest)
        if restored!=graph:raise ValueError('Stored adapter graph codec differs')
        kwargs['cdg']=restored
        result=await original(session,*args,**kwargs)
        for item in result['trace']:
            runtime=nodes[item['node_id']].matched_primitive
            if runtime in stored:seen.add(runtime)
        traces.append(dict(graph_sha256=digest,executed_nodes=len(result['trace']),status=result['status']))
        return result

    with patch.object(runner.CDGExecutionSession,'execute',execute):
        raw=raw_validate(source)
        wiring=wiring_validate()
        native=state_validate(source,checkpoint)
    required={a['runtime_fqdn'] for a in proposed['atoms'] if a['role']=='domain_adapter'}
    if required-seen:raise ValueError('Unexecuted stored domain adapters: '+str(sorted(required-seen)))
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,native_threads=1,
        stored_domain_adapters=len(required),executed_stored_providers=len(seen),
        exact_prepared_rows=raw['exact_prepared_rows'],native_prediction_comparisons=native['exact_native_query_comparisons'],
        recorded_training_fit_invocations=wiring['fit_invocations'],
        bindings={runtime:{key:stored[runtime]['atom'][key] for key in ['version_id','content_hash']} for runtime in sorted(seen)},
        execution_sessions=traces,
        implementation_sha256={name:sha(ROOT/name) for name in ['scripts/validate_lifecycle_stored_adapters.py',
            'scripts/validate_nasa_first_raw_training_graph.py','scripts/validate_nasa_first_population_training_wiring.py',
            'scripts/validate_nasa_first_policy_state_graphs.py','sciona/visualizer/runner.py','sciona/services/execution_graph_codec.py']},
        limitations=['Topology is built locally from qualified graphs and stored provider ports; served-CDG retrieval remains pending.',
                     'Training section uses its recording fit backend; native source-sized fitting retains separate qualification.',
                     'All nineteen domain adapters execute normally; native inference uses qualified saved models.',
                     'Intermediate persistence uses existing qualification capture hooks.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    parser.add_argument('--checkpoint-directory',type=Path,required=True);args=parser.parse_args()
    report=validate(args.source_directory,args.checkpoint_directory)
    (ROOT/'docs/reviews/competition_nasa_first_lifecycle_stored_adapters.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['passed','stored_domain_adapters','executed_stored_providers','native_prediction_comparisons']}))
