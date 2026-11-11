"""Actual serialized graph/provider execution with synthetic runtime payload."""
import asyncio
import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import sciona.atoms.dl.plasticc_execution as provider
from sciona.plasticc_graph import build_plasticc_graph
from sciona.plasticc_runtime import prepare_runtime,train_runtime,predict_runtime
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.visualizer import runner
from validate_plasticc_runtime import synthetic_payload


def run_graph(graph,payload):
    captured={}
    def capture(directory,node,name,value):
        if name.startswith('out_'): captured[(node,name[4:])]=value
    with tempfile.TemporaryDirectory(prefix='synthetic-plasticc-') as directory:
        with patch.object(runner,'RUNS_DIR',Path(directory)),patch.object(runner,'save_intermediate_value',side_effect=capture):
            result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-plasticc','case').execute({'payload':payload},cdg=graph))
    return result,captured


def validate():
    digest,nodes,edges=encode_execution_graph(build_plasticc_graph())
    graph=decode_execution_graph(nodes,edges,digest)
    assert len(graph.nodes)==3 and len(graph.edges)==2
    witness={'payload':{}}
    for node in graph.nodes:
        fn=getattr(provider,node.matched_primitive.rsplit('.',1)[-1])
        assert list(inspect.signature(fn).parameters)==[p.name for p in node.inputs]
        witness[node.outputs[0].name]=getattr(provider,'witness_'+fn.__name__)(**{p.name:witness[p.name] for p in node.inputs})
    payload=synthetic_payload(); original=copy.deepcopy(payload)
    direct=predict_runtime(train_runtime(prepare_runtime(payload)))
    outcome,captured=run_graph(graph,json.loads(json.dumps(payload,allow_nan=False)))
    assert outcome['status']=='completed'
    assert captured[('predict','result')]==direct and payload==original
    counts=dict(provider_contracts=3,serialized_graph_cases=1,parameter_rejections=0,runner_rejections=0)
    for key,value in [('n_estimators',0),('n_estimators',1.5),('num_leaves',1),('n_jobs',0),
                      ('max_depth',0),('random_seed',-1),('reg_alpha',-1),('learning_rate',0),('colsample_bytree',1.1)]:
        bad=copy.deepcopy(payload); bad['config']['training_parameters'][key]=value
        try: prepare_runtime(bad)
        except ValueError: counts['parameter_rejections']+=1
        else: raise AssertionError('Invalid training parameter accepted')
    bad=copy.deepcopy(payload); bad['version']=True
    try: run_graph(graph,bad)
    except RuntimeError as error:
        assert 'plasticc_prepare' in str(error)
        counts['runner_rejections']+=1
    else: raise AssertionError('Runner accepted invalid payload')
    paths=[Path(__file__),ROOT/'scripts/validate_plasticc_runtime.py']+sorted((ROOT/'sciona').glob('plasticc_*.py'))
    return dict(approved=False,synthetic_only=True,checks=counts,serialized_graph_sha256=digest,
        provider_sha256=hashlib.sha256(Path(inspect.getfile(provider)).read_bytes()).hexdigest(),
        implementation_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        limitations=['Local serialized graph execution, not served catalog retrieval',
                     'Automated semantic/provenance review and transactional publication outstanding'])

if __name__=='__main__':
    report=validate()
    (ROOT/'docs/reviews/competition_plasticc_graph.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks'],sort_keys=True))
