"""Structural checks only: combined graph wiring and coordinate routing contract."""
import hashlib
import importlib
import inspect
import json
from pathlib import Path

from sciona.architect.models import EdgeKind
from sciona.osic_combined_inference_graph import build_combined_inference_graph
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph
from sciona.atoms.ml.domain_adapters import osic_coordinate_routing as provider
from sciona.ghost.registry import REGISTRY

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate():
    for first,second in [('synthetic-metre-per-second','synthetic-metre'),
                         ('synthetic-watt-per-second','synthetic-watt'),(' same ',' same ')]:
        result=provider.route_response_coordinates(first,second)
        assert type(result) is tuple and result==(first,second)
        assert result[0] is first and result[1] is second
    class StringSubclass(str):pass
    rejected=0
    for bad in ('', ' ', None, 1, StringSubclass('coordinate')):
        for position in (0,1):
            args=['first','second'];args[position]=bad
            try:provider.route_response_coordinates(*args)
            except ValueError:rejected+=1
            else:raise AssertionError('Invalid coordinate declaration accepted')
    try:provider.witness_coordinates()
    except ValueError:rejected+=1
    else:raise AssertionError('Abstract coordinate routing accepted')
    graph=build_combined_inference_graph()
    digest,nodes,edges=encode_execution_graph(graph)
    restored=decode_execution_graph(nodes,edges,digest)
    assert encode_execution_graph(restored)==(digest,nodes,edges)
    assert len(restored.nodes)==15 and len(restored.edges)==26
    assert len({n.node_id for n in restored.nodes})==15
    for index in range(len(edges)):
        try:decode_execution_graph(nodes,edges[:index]+edges[index+1:],digest)
        except ValueError:pass
        else:raise AssertionError('Missing edge accepted')
    hashes={}
    for node in restored.nodes:
        module,symbol=node.matched_primitive.rsplit('.',1)
        imported=importlib.import_module(module)
        fn=getattr(imported,symbol)
        assert REGISTRY[node.matched_primitive]['impl'] is fn
        assert list(inspect.signature(fn).parameters)==[p.name for p in node.inputs]
        hashes[module]=sha(inspect.getfile(imported))
    incoming={(e.target_id,e.input_name):e for e in restored.edges}
    assert len(incoming)==26
    for target,port in [('image_restore','image'),('quantile_features','quantile'),('quantile_replay','quantile')]:
        edge=incoming[target,'response_coordinate']
        assert (edge.source_id,edge.output_name)==('coordinates',port)
    callable_sources={e.source_id for e in restored.edges if e.edge_kind==EdgeKind.CALLABLE_INJECTION}
    assert callable_sources=={'image_prepare','image_select_inputs','image_predict'}
    roots={p.name for n in restored.nodes if n.node_id not in callable_sources
           for p in n.inputs if (n.node_id,p.name) not in incoming}
    assert roots=={'directory','image_coordinate','metadata_coordinate','baselines','queries','planes',
                   'batch_size','state','states','dtype','feature_coordinate',
                   'image_response_coordinate','quantile_response_coordinate'}
    # Shared keyed rows feed image routing and the quantile key split. Quantile
    # feature replay receives only the split rows through its authoritative edge.
    assert ('image_population','queries') not in incoming
    assert ('quantile_query_route','queries') not in incoming
    edge=incoming['quantile_features','queries']
    assert (edge.source_id,edge.output_name)==('quantile_query_route','rows')
    files=['sciona/osic_combined_inference_graph.py','sciona/osic_image_population_graph.py',
           'sciona/osic_image_slice_graph.py','sciona/osic_image_lifecycle_graphs.py',
           'sciona/osic_keyed_quantile_graph.py','sciona/osic_quantile_replay_graph.py',
           'sciona/services/execution_graph_codec.py','scripts/validate_osic_combined_graph_structure.py']
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,
        graph_sha256=digest,nodes=15,edges=26,missing_edges_rejected=26,
        coordinate_cases=3,coordinate_rejections=rejected,independent_coordinate_edges=True,
        exact_registry_signatures=True,root_inputs=sorted(roots),
        callable_sources=sorted(callable_sources),shared_keyed_queries=True,
        provider_sha256=hashes,software_sha256={name:sha(ROOT/name) for name in files},
        limitations=['Structure and coordinate string routing only; combined numerical execution has not been qualified.',
            'Coordinate declarations are opaque caller labels; no unit compatibility or conversion is inferred.',
            'Coordinate provider is not catalog-approved. Graph publication and full original workflow remain pending.',
            'No empirical effectiveness, HTTP transport or full-training claim.'])


if __name__=='__main__':
    result=validate()
    (ROOT/'docs/reviews/osic_combined_graph_structure.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,nodes=result['nodes'],edges=result['edges'],numerical_execution=False)))
