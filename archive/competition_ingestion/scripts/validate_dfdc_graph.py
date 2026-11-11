"""Actual serialized DFDC provider graph on the complete synthetic runtime case."""
import asyncio
import numpy as np
import hashlib
import inspect
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import sciona.atoms.dl.dfdc_execution as provider
from sciona.dfdc_graph import build_dfdc_graph
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.visualizer import runner
import validate_dfdc_runtime_full as full


def run_graph(graph,payload):
    captured={}
    def capture(directory,node,name,value):
        if node=='execute' and name=='out_result':captured['result']=value
    with tempfile.TemporaryDirectory(prefix='sciona-synthetic-dfdc-graph-') as directory:
        with patch.object(runner,'RUNS_DIR',Path(directory)),patch.object(runner,'save_intermediate_value',side_effect=capture):
            status=asyncio.run(runner.CDGExecutionSession(None,'synthetic-dfdc','case').execute({'payload':payload},cdg=graph))
    assert status['status']=='completed'
    return captured['result']


def main():
    # Provider discovery imports all configured packages, some of which initialize
    # NumPy state. Measure that startup effect separately from steady execution.
    before=np.random.get_state()
    runner._ensure_atoms_imported()
    after=np.random.get_state()
    discovery_changed_numpy=not (before[0]==after[0] and np.array_equal(before[1],after[1]) and before[2:]==after[2:])
    runner._ensure_atoms_imported()
    repeated=np.random.get_state()
    assert after[0]==repeated[0] and np.array_equal(after[1],repeated[1]) and after[2:]==repeated[2:]
    digest,nodes,edges=encode_execution_graph(build_dfdc_graph())
    graph=decode_execution_graph(nodes,edges,digest)
    assert len(graph.nodes)==2 and len(graph.edges)==1
    witness={'payload':{}}
    for node in graph.nodes:
        fn=getattr(provider,node.matched_primitive.rsplit('.',1)[-1])
        assert list(inspect.signature(fn).parameters)==[p.name for p in node.inputs]
        witness[node.outputs[0].name]=getattr(provider,'witness_'+fn.__name__)(**{p.name:witness[p.name] for p in node.inputs})
    assert witness['result']=={'kind':'DFDC.Result'}
    results=[]
    def execute(payload):
        result=run_graph(graph,payload);results.append(result);return result
    # Execute every full runtime assertion via the real graph runner/providers.
    with patch.object(full,'execute',side_effect=execute):full.main()
    assert len(results)==1 and len(results[0]['runs'])==5
    try:run_graph(graph,{'version':True})
    except RuntimeError:rejected=True
    else:raise AssertionError('invalid graph payload accepted')
    paths=[str(p.relative_to(ROOT)) for p in sorted((ROOT/'sciona').glob('dfdc_*.py'))]
    paths+=['scripts/validate_dfdc_graph.py','scripts/validate_dfdc_runtime_full.py']
    report={'format':'dfdc-graph-validation.v1','approved':False,'result':'passed',
            'checks':{'provider_witness_contracts':2,'serialized_full_runtime_graphs':1,
                      'actual_B7_training_runs':5,'training_updates':10,'ensemble_states':7,'runner_rejections':int(rejected)},
            'provider_discovery_changed_numpy_on_startup':discovery_changed_numpy,
            'serialized_graph_sha256':digest,
            'provider_sha256':hashlib.sha256(Path(inspect.getfile(provider)).read_bytes()).hexdigest(),
            'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
            'limits':'Actual runner/providers and complete native/neural synthetic runtime, using explicit shortened plan and random B7 initialization. Provider discovery is initialized before the caller-RNG baseline; cold package imports may change NumPy state. No served catalog, historical checkpoint identity or default-scale quality claim. Intermediate private values captured only in memory during this synthetic validation.'}
    (ROOT/'docs/reviews/competition_dfdc_graph.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
