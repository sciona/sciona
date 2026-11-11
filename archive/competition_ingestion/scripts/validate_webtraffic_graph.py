"""Versioned synthetic payload and actual serialized CDG runner validation."""
import argparse
import asyncio
import copy
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import torch
import sciona.atoms.dl.webtraffic_execution as provider
from sciona.webtraffic_graph import build_webtraffic_graph
from sciona.webtraffic_payload import decode_payload,execute_payload
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.visualizer import runner


def synthetic_payload():
    rng=np.random.RandomState(1294)
    pages=sorted(f'SyntheticGraph{i}_{site}_{agent}' for i,site in enumerate(
        ['en.wikipedia.org','fr.wikipedia.org','commons.wikimedia.org','www.mediawiki.org'])
        for agent in ['all-access_spider','desktop_all-agents','mobile-web_all-agents','all-access_all-agents'])
    counts=rng.poisson(np.arange(4,20)[:,None],size=(16,800)).astype(float);counts[0]=0;counts[1,0]=np.nan
    return dict(version=1,pages=pages,start_day='2000-01-01',counts=np.where(np.isnan(counts),None,counts).tolist(),
                config=dict(seed=2,batch_size=2,max_steps=1,save_from_step=1,prediction_batch_size=7))


def run_graph(graph,payload):
    captured={}
    def capture(directory,node,name,value):
        if name.startswith('out_'):captured[(node,name[4:])]=value
    with tempfile.TemporaryDirectory(prefix='synthetic-webtraffic-graph-') as directory:
        with patch.object(runner,'RUNS_DIR',Path(directory)),patch.object(runner,'save_intermediate_value',side_effect=capture):
            result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-webtraffic','case').execute({'payload':payload},cdg=graph))
    return result,captured


def validate(root,source):
    torch.set_num_threads(1)
    pins=json.loads((root/'docs/reviews/competition_webtraffic_source_pins.json').read_text())
    assert hashlib.sha256((source/'model.py').read_bytes()).hexdigest()==pins['files']['model.py']
    digest,nodes,edges=encode_execution_graph(build_webtraffic_graph())
    graph=decode_execution_graph(nodes,edges,digest)
    assert len(graph.nodes)==3 and len(graph.edges)==2
    counts=dict(provider_contracts=0,serialized_graph_cases=0,payload_rejections=0,runner_rejections=0)
    witness={'payload':{}}
    for node in graph.nodes:
        fn=getattr(provider,node.matched_primitive.rsplit('.',1)[-1])
        assert list(inspect.signature(fn).parameters)==[p.name for p in node.inputs]
        w=getattr(provider,'witness_'+fn.__name__)
        witness[node.outputs[0].name]=w(**{p.name:witness[p.name] for p in node.inputs})
        counts['provider_contracts']+=1
    payload=synthetic_payload();wire=json.loads(json.dumps(payload,allow_nan=False));original=copy.deepcopy(wire)
    direct=execute_payload(wire)
    outcome,captured=run_graph(graph,wire)
    assert outcome['status']=='completed',outcome.get('error')
    assert captured[('forecast','result')]==direct and wire==original
    json.dumps(direct,allow_nan=False)
    assert len(direct['days'])==63 and len(direct['forecasts'])==16
    counts['serialized_graph_cases']+=1
    cases=[]
    for kind in ['version','duplicate_page','boolean_count','negative_count','unequal_rows','unknown_config','invalid_date']:
        bad=copy.deepcopy(wire)
        if kind=='version':bad['version']=True
        elif kind=='duplicate_page':bad['pages'][1]=bad['pages'][0]
        elif kind=='boolean_count':bad['counts'][1][1]=True
        elif kind=='negative_count':bad['counts'][1][1]=-1
        elif kind=='unequal_rows':bad['counts'][1].pop()
        elif kind=='unknown_config':bad['config']['unknown']=1
        else:bad['start_day']='not-a-date'
        cases.append(bad)
        try:decode_payload(bad)
        except (ValueError,TypeError):counts['payload_rejections']+=1
        else:raise AssertionError('Invalid wire payload accepted')
    try:
        run_graph(graph,cases[2])
    except RuntimeError as error:
        assert 'webtraffic_prepare' in str(error) and 'Nonnegative finite counts or null' in str(error)
    else:raise AssertionError('Runner accepted invalid payload')
    counts['runner_rejections']+=1
    paths=['scripts/validate_webtraffic_graph.py']+[str(p.relative_to(root)) for p in sorted((root/'sciona').glob('webtraffic_*.py'))]
    paths+=['docs/reviews/competition_webtraffic_source_pins.json']
    return dict(approved=False,synthetic_only=True,checks=counts,serialized_graph_sha256=digest,
                provider_sha256=hashlib.sha256(Path(inspect.getfile(provider)).read_bytes()).hexdigest(),
                implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
                limitations=['Derived Community RNG/EMA contracts are explicit; no original TensorFlow race/seeded-run claim.',
                             'Three-node full active lifecycle; original intake remains untouched.',
                             'Local serialized graph execution, not served catalog retrieval or approval.',
                             'Automated semantic review and idempotent publication verification remain.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_webtraffic_graph.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'implementation_sha256','provider_sha256'}}))
