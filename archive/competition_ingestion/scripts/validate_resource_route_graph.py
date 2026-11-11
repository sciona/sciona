"""Exercise all five route stages through the actual serialized graph runner."""
import asyncio
import hashlib
import inspect
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from resource_route_synthetic import payload
from sciona.resource_route_graph import build_resource_route_graph
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.visualizer import runner
import sciona.atoms.ml.resource_route_execution as provider


def main():
    digest,nodes,edges=encode_execution_graph(build_resource_route_graph())
    graph=decode_execution_graph(nodes,edges,digest)
    assert encode_execution_graph(graph)[0]==digest
    runner._ensure_atoms_imported()
    cases=[]
    normal=payload();cases.append((normal,'optimal',['detour','join','finish'],3))
    blocked=payload();blocked['blocked_edges']=['detour'];cases.append((blocked,'infeasible',[],None))
    limited=payload();limited['max_expansions']=1;cases.append((limited,'search_limit',[],None))
    identity=payload();identity['goal']='s';cases.append((identity,'optimal',[],0))
    for index,(data,expected_status,expected_path,expected_cost) in enumerate(cases):
        captured={};executed=set()
        def capture(directory,node,name,value):
            executed.add(node)
            if node=='select' and name=='out_result':captured['result']=value
        with tempfile.TemporaryDirectory(prefix='route-synthetic-') as temporary:
            with patch.object(runner,'RUNS_DIR',Path(temporary)),patch.object(runner,'save_intermediate_value',side_effect=capture):
                result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-resource-route',str(index)).execute({'payload':data},cdg=graph))
        assert result['status']=='completed',result['status']
        values=json.loads(json.dumps(captured['result'],allow_nan=False))
        assert values['status']==expected_status and values['edge_ids']==expected_path and values['cost']==expected_cost
        assert executed=={'encode','generate','search','validate','select'},executed
    witness={}
    for name in ('encode','generate','search','validate','select'):
        witness=getattr(provider,'witness_resource_route_'+name)(witness)
    assert witness=={'kind':'ResourceRoute.Result'}
    paths=['sciona/resource_route_planning.py','sciona/resource_route_graph.py','scripts/resource_route_synthetic.py','scripts/validate_resource_route_graph.py']
    report=dict(status='passed',approved=False,serialized_graph_sha256=digest,
        provider_sha256=hashlib.sha256(Path(inspect.getfile(provider)).read_bytes()).hexdigest(),
        code_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
        checks=dict(actual_runner_nodes=5,serialized_edges=4,executed_cases=4,optimal=True,infeasible=True,search_limit=True,identity_route=True,witness_chain=True,strict_json_output=True),
        limits='Synthetic serialized execution only. No catalog approval or historical competition claim.')
    (ROOT/'docs/reviews/competition_resource_route_graph_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
