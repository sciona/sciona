"""Qualify explicit quantile replay through the canonical graph and real runner."""
from scripts import validate_osic_raw_table_replay as reference
import argparse
import asyncio
import importlib
import inspect
import json
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch

from sciona.osic_quantile_replay_graph import build_quantile_replay_graph
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph

ROOT=reference.ROOT


def child(directory):
    from sciona.ghost.registry import REGISTRY
    from sciona.visualizer import runner
    from sciona.atoms.ml.domain_adapters import osic_feature_projection as projection
    def forbidden(*args,**kwargs):raise AssertionError('Replay attempted fitting')
    for name in ('fit','train_step','train_on_batch'):setattr(reference.keras.Model,name,forbidden)
    reference.keras.optimizers.legacy.Adam.apply_gradients=forbidden
    for module,names in ((reference.features,('prepare_features','assemble_populations','joint_population_features')),
        (reference.feature_state,('export_preparation_state','assemble_populations','joint_population_features')),
        (reference.training,('prepare_features','train_quantile_branch','train_prepared_quantile_folds'))):
        for name in names:setattr(module,name,forbidden)
    bundle=json.loads((directory/'state.json').read_text())
    queries=reference.pd.DataFrame(json.loads((directory/'queries.json').read_text()))
    digest,nodes,edges=encode_execution_graph(build_quantile_replay_graph())
    graph=decode_execution_graph(nodes,edges,digest)
    for index in range(len(edges)):
        try:decode_execution_graph(nodes,edges[:index]+edges[index+1:],digest)
        except ValueError:pass
        else:raise ValueError('Missing graph edge accepted')
    provider_hashes={}
    def discover():
        for node in graph.nodes:
            module,symbol=node.matched_primitive.rsplit('.',1)
            imported=importlib.import_module(module);fn=getattr(imported,symbol)
            if REGISTRY[node.matched_primitive]['impl'] is not fn:raise ValueError('Registry implementation differs')
            if list(inspect.signature(fn).parameters)!=[p.name for p in node.inputs]:raise ValueError('Signature differs')
            provider_hashes[module]=reference.sha(Path(inspect.getfile(imported)))
    captured={}
    def capture(path,node,name,value):
        if name.startswith('out_'):captured[node,name[4:]]=value
    payload=dict(state=bundle['preparation'],queries=queries,states=bundle['models'],dtype='float32',
                 feature_coordinate=reference.FEATURE,response_coordinate=reference.RESPONSE)
    with tempfile.TemporaryDirectory(prefix='sciona-quantile-graph-',dir='/private/tmp') as temporary, \
         patch.object(runner,'RUNS_DIR',Path(temporary)),patch.object(runner,'save_intermediate_value',capture), \
         patch.object(runner,'_ensure_atoms_imported',discover):
        result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-quantile-replay','graph').execute(payload,cdg=graph))
        trace=json.loads((Path(temporary)/'graph/execution_trace.json').read_text())
    if result['status']!='completed' or len(captured)!=9:raise ValueError('Complete quantile graph execution required')
    with reference.np.load(directory/'expected.npz',allow_pickle=False) as expected:
        reference.np.testing.assert_array_equal(captured['project','features'],expected['features'])
        reference.np.testing.assert_array_equal(captured['reduce','result'],expected['predictions'])
    for port,key in [('features','features'),('columns','feature_columns'),('mask','observed_mask'),('values','observed_values')]:
        if captured['project',port] is not captured['features','result'][key]:raise ValueError('Projection must borrow values')
    assert len(trace)==6 and not any(row['cached'] for row in trace)
    for bad in ({},dict(features=1,feature_columns=2,observed_mask=3,observed_values=4,extra=5)):
        try:projection.unpack_quantile_features(bad)
        except ValueError:pass
        else:raise ValueError('Invalid projection accepted')
    try:projection.witness_projection()
    except ValueError:pass
    else:raise ValueError('Abstract projection accepted')
    return dict(passed=True,raw_queries=len(queries),feature_values=int(captured['project','features'].size),
        features_exact=True,predictions_exact=True,feature_fitting_disabled=True,model_updates_disabled=True,
        graph_execution=dict(graph_sha256=digest,nodes=6,edges=8,missing_edges_rejected=8,
            projection_borrows_values=True,projection_rejections=3,node_order=[r['node_id'] for r in trace],
            provider_sha256=provider_hashes))


def validate():
    paths=['sciona/osic_quantile_replay_graph.py','scripts/validate_osic_quantile_replay_graph.py',
           'sciona/visualizer/runner.py','sciona/services/execution_graph_codec.py']
    before={name:reference.sha(ROOT/name) for name in paths}
    run=subprocess.run;redirects=0
    def redirect(command,*args,**kwargs):
        nonlocal redirects
        command=list(command)
        target=str(Path(reference.__file__).resolve())
        if target in command and '--replay-directory' in command:
            command[command.index(target)]=str(Path(__file__).resolve());redirects+=1
        return run(command,*args,**kwargs)
    with patch.object(reference.subprocess,'run',redirect):report=reference.validate()
    graph=report['fresh_process_replay'].pop('graph_execution')
    baseline=json.loads((ROOT/'docs/reviews/osic_raw_table_replay.json').read_text())
    if report!=baseline:raise ValueError('Graph differs from qualified raw replay')
    if redirects!=1 or before!={name:reference.sha(ROOT/name) for name in paths}:raise ValueError('Graph inputs changed')
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,graph_execution=graph,
        software_sha256=before,raw_replay_qualification=report,
        limitations=['Graph executes generated query rows and two freshly trained 600-epoch synthetic fold states; no empirical effectiveness claim.',
            'Projection and conversion providers are locally qualified, not catalog-approved; graph and original workflow remain unapproved.',
            'Explicit provider discovery and intercepted intermediate persistence; HTTP transport, default discovery and caching remain unqualified.',
            'Query-key extraction and final image/quantile blend remain separate outstanding graph work.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--replay-directory',type=Path)
    args=parser.parse_args()
    if args.replay_directory:print(json.dumps(child(args.replay_directory)))
    else:
        result=validate()
        (ROOT/'docs/reviews/osic_quantile_replay_graph.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(passed=True,nodes=6,edges=8,exact_predictions=True,approved=False)))
