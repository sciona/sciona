"""Key-preserving query routing into the explicit frozen quantile graph."""
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus
from sciona.osic_quantile_replay_graph import build_quantile_replay_graph


def build_keyed_quantile_graph():
    graph=build_quantile_replay_graph()
    graph.nodes.insert(0, AlgorithmicNode(
        node_id='query_route', name='Preserve query keys and coordinates',
        description='Split supplied keyed rows without sorting, key generation or numerical conversion.',
        concept_type='custom', status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.ml.domain_adapters.osic_query_routing.split_keyed_queries',
        inputs=[IOSpec(name='queries',type_desc='pandas.DataFrame',constraints=
            'Exact key/entity/time columns; unique nonblank keys, nonblank entities, native int64 times. Caller owns meanings and frozen scope agreement.')],
        outputs=[IOSpec(name='rows',type_desc='pandas.DataFrame',constraints=
            'Copied entity/time rows in supplied order with original index and dtype.'),
            IOSpec(name='keys',type_desc='list[str]',constraints=
            'Supplied keys in exactly the same order as rows and final predictions.')]))
    graph.edges.append(DependencyEdge(source_id='query_route',target_id='features',
        output_name='rows',input_name='queries',source_type='pandas.DataFrame',target_type='pandas.DataFrame'))
    graph.metadata.update(num_nodes=7,num_edges=9,
        scope='Keyed query routing and frozen quantile inference with aligned explicit boundary outputs.',
        boundary_outputs=[dict(node='query_route',port='keys'),dict(node='reduce',port='result')],
        exclusions=['Image branch and final blending','Full training and original-CDG approval',
                    'HTTP transport and empirical effectiveness'])
    return graph
