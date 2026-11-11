"""Run a decoded prepared-image graph against frozen synthetic numeric replay."""
import argparse
import asyncio
import importlib
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from unittest.mock import patch

from sciona.osic_image_lifecycle_graphs import build_prepared_image_inference_graph
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph
from scripts import validate_osic_image_state as original
from scripts.import_flavours_provider_drafts import connect

RUN = subprocess.run


def redirect(command,*args,**kwargs):
    command=list(command)
    target=str(Path(original.__file__).resolve())
    if target in command and any(mode in command for mode in ('--worker','--replay')):
        command[command.index(target)]=str(Path(__file__).resolve())
    return RUN(command,*args,**kwargs)


def replay(directory):
    keras,np=original.runtime()
    from sciona.ghost.registry import REGISTRY
    from sciona.visualizer import runner
    graph=build_prepared_image_inference_graph()
    digest,nodes,edges=encode_execution_graph(graph)
    graph=decode_execution_graph(nodes,edges,digest)
    try:
        decode_execution_graph(nodes,[],digest)
    except ValueError:
        pass
    else:
        raise ValueError('Missing model-state edge was accepted')
    reviews=[json.loads((original.ROOT/'docs/reviews'/name).read_text()) for name in
             ('osic_image_lifecycle_semantics.json','keras_bounded_semantics.json')]
    records={p['fqdn']:(p,r['software_sha256']['provider']) for r in reviews for p in r['providers']}
    bindings=[]
    with connect() as db:
        db.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        for node in graph.nodes:
            record,provider_hash=records[node.matched_primitive]
            module,symbol=node.matched_primitive.rsplit('.',1)
            imported=importlib.import_module(module)
            if original.sha(inspect.getfile(imported))!=provider_hash:
                raise ValueError('Approved provider implementation changed')
            expected=dict(fqdn=record['fqdn'],version_id=record['version_id'],
                content_hash=record['content_hash'],trust_tier=3,import_module=module,source_symbol=symbol)
            for prefix,key in [('atom','atom_id'),('artifact','artifact_id')]:
                rows=db.execute(f'''SELECT s.fqdn,v.version_id::text,v.content_hash,v.trust_tier,
                    a.import_module,a.source_symbol FROM catalog_{prefix}s_served s
                    JOIN {prefix}_versions v USING({key}) JOIN atoms a ON a.atom_id=s.{key}
                    WHERE s.{key}=%s AND v.is_latest''',(record['artifact_id'],)).fetchall()
                if not rows or any(row!=expected for row in rows):
                    raise ValueError('Exact approved served bindings required')
            fn=getattr(imported,symbol)
            if REGISTRY[node.matched_primitive]['impl'] is not fn:
                raise ValueError('Execution registry differs from served implementation')
            if list(inspect.signature(fn).parameters)!=[p.name for p in node.inputs]:
                raise ValueError('Graph inputs differ from callable signature')
            bindings.append(expected)
    def forbidden(*args,**kwargs):
        raise AssertionError('Fitting and HDF5 loading are forbidden during graph replay')
    for name in ('fit','train_step','train_on_batch'):
        setattr(keras.Model,name,forbidden)
    keras.optimizers.legacy.Adam.apply_gradients=forbidden
    keras.models.load_model=forbidden
    captured={}
    def capture(path,node,name,value):
        if name=='out_result':captured[node]=value
    def explicit_discovery():
        if any(node.matched_primitive not in REGISTRY for node in graph.nodes):
            raise ValueError('Explicitly imported graph provider missing')
    with np.load(directory/'queries.npz',allow_pickle=False) as values:
        inputs=[values['image'],values['metadata']]
        payload=dict(directory=directory/'state',inputs=inputs,batch_size=32,**original.COORDINATES)
        with patch.object(runner,'RUNS_DIR',directory/'runs'), \
             patch.object(runner,'save_intermediate_value',side_effect=capture), \
             patch.object(runner,'_ensure_atoms_imported',explicit_discovery):
            result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-image-inference','fresh').execute(payload,cdg=graph))
        if result['status']!='completed' or set(captured)!={'restore','predict'}:
            raise ValueError('Complete restored-model graph execution required')
        if getattr(captured['restore'],'optimizer',None) is not None:
            raise ValueError('Inference graph restored optimizer state')
        np.testing.assert_array_equal(captured['predict'],values['expected'])
    trace=json.loads((directory/'runs/fresh/execution_trace.json').read_text())
    if [r['node_id'] for r in trace]!=['restore','predict'] or any(r['cached'] for r in trace):
        raise ValueError('Expected uncached sequential graph execution')
    return dict(passed=True,queries=3,exact_predictions=True,fitting_disabled=True,
        hdf5_loading_disabled=True,optimizer_state_absent=True,
        graph_execution=dict(graph_sha256=digest,nodes=2,edges=1,decoded_projection=True,
            missing_edge_rejected=True,bindings=bindings,uncached_node_order=['restore','predict'],
            explicit_provider_discovery=True,intermediate_persistence_disabled=True))


def validate(dependencies,source_directory):
    files=['sciona/osic_image_lifecycle_graphs.py','sciona/services/execution_graph_codec.py',
           'sciona/visualizer/runner.py','scripts/validate_osic_prepared_inference_graph.py']
    software={name:original.sha(original.ROOT/name) for name in files}
    graph_digest=encode_execution_graph(build_prepared_image_inference_graph())[0]
    source_directory=Path(source_directory)
    sources=[source_directory/name for name in ('source.h5','snapshot.json')]
    before=[original.sha(path) for path in sources]
    with tempfile.TemporaryDirectory(prefix='sciona-prepared-image-graph-',dir='/private/tmp') as temporary:
        directory=Path(temporary)
        for path in sources:shutil.copyfile(path,directory/path.name)
        with patch.object(original.subprocess,'run',redirect):
            report=original.validate(dependencies,directory)
    if before!=[original.sha(path) for path in sources]:
        raise ValueError('Private source checkpoint changed')
    execution=report['execution']['fresh_process_replay'].pop('graph_execution')
    baseline=json.loads((original.ROOT/'docs/reviews/osic_image_state.json').read_text())
    if report!=baseline:raise ValueError('Decoded graph differs from frozen numeric replay')
    if (software!={name:original.sha(original.ROOT/name) for name in files}
            or execution['graph_sha256']!=graph_digest):
        raise ValueError('Graph execution inputs changed during qualification')
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,
        graph_execution=execution,state_qualification=report,private_source_unchanged=True,
        software_sha256=software,
        limitations=['Prepared arrays and numeric model state only; raw preparation, population routing, reduction, blend and training are separate outstanding graph work.',
            'Graph providers are already served at Tier 3, but this graph is not catalog-approved and does not approve the original CDG.',
            'The real sequential graph executor runs with explicit provider discovery and intercepted intermediate persistence; HTTP input transport, default provider discovery and visualizer caching are not qualified.',
            'Intermediate synthetic checkpoint only; exact predictions do not establish empirical effectiveness or complete training.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-directory',type=Path,required=True)
    parser.add_argument('--dependencies',type=Path)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--worker',action='store_true')
    modes.add_argument('--replay',action='store_true')
    args=parser.parse_args()
    if args.replay:print(json.dumps(replay(args.runtime_directory)))
    elif args.worker:
        with patch.object(original.subprocess,'run',redirect):
            result=original.worker(args.runtime_directory)
        print(json.dumps(result))
    else:
        if not args.dependencies:parser.error('--dependencies required')
        result=validate(args.dependencies,args.runtime_directory)
        (original.ROOT/'docs/reviews/osic_prepared_inference_graph.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(passed=True,nodes=2,edges=1,exact_fresh_replay=True,approved=False)))
