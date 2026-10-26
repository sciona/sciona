"""Draft explicit meta-stage graph; not the complete Otto source workflow."""
import inspect

from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus


def build_meta_graph():
    import sciona.atoms.ml.stacking_meta as provider
    nodes, edges = [], []

    def node(name, function, output, output_type='object'):
        fn = getattr(provider, function)
        inputs = [IOSpec(name=k, type_desc=p.annotation.__name__)
                  for k, p in inspect.signature(fn).parameters.items()]
        nodes.append(AlgorithmicNode(node_id=name, name=name,
            description=(fn.__doc__ or function).strip(), concept_type='custom',
            status=NodeStatus.ATOMIC, matched_primitive=fn.__module__+'.'+fn.__name__,
            inputs=inputs, outputs=[IOSpec(name=output, type_desc=output_type)]))

    def edge(source, output, target, port, kind='object'):
        edges.append(DependencyEdge(source_id=source, target_id=target,
            output_name=output, input_name=port, source_type=kind, target_type=kind))

    node('prepare', 'prepare_meta', 'inputs')
    for family, select in (('xgboost', 'select_xgboost'), ('neural', 'select_neural'),
                           ('extratrees', 'select_extratrees')):
        node('select_'+family, select, 'selection')
        node('refit_'+family, 'refit_meta', 'bag')
        node('mean_'+family, 'mean_probability_bag', 'mean')
        edge('prepare', 'inputs', 'select_'+family, 'inputs')
        edge('prepare', 'inputs', 'refit_'+family, 'inputs')
        edge('select_'+family, 'selection', 'refit_'+family, 'selection')
        edge('refit_'+family, 'bag', 'mean_'+family, 'values')
    node('blend', 'blend_meta', 'blend', 'dict')
    node('result', 'meta_result', 'result', 'dict')
    for family in ('xgboost', 'neural', 'extratrees'):
        edge('mean_'+family, 'mean', 'blend', family)
        edge('select_'+family, 'selection', 'result', family)
    edge('blend', 'blend', 'result', 'blend', 'dict')
    return CDGExport(nodes=nodes, edges=edges, metadata={
        'publication_status': 'draft', 'num_nodes': len(nodes), 'num_edges': len(edges),
        'scope': 'Meta-only source-policy adapter: separate selection, complete refits, reusable means and final blend.',
        'source_version_ids': ['dc4969f7-e5a4-5644-a9d0-fa10493458ed'],
        'limitations': ['Requires caller-supplied first-level and supplemental feature matrices.',
                       'Nine-class four-fold source policy; not a complete source-workflow qualification.',
                       'Provisioned in-process execution; no cached-state or static-shape qualification.',
                       'Selection losses do not independently estimate generalization.']})
