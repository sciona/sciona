"""Draft first-level graph exposing each source entry and shared KNN groups."""
import inspect

from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus


def build_first_graph(knn_groups=None):
    import sciona.atoms.ml.stacking_first as provider
    groups = [list(range(24, 34))] if knn_groups is None else knn_groups
    if (type(groups) is not list or not groups
            or any(type(g) is not list or not g or any(type(i) is not int for i in g) for g in groups)
            or sorted(i for g in groups for i in g) != list(range(24, 34))):
        raise ValueError('Neighbor groups must partition source entries 24 through 33')
    nodes, edges = [], []

    def node(name, function, outputs):
        fn = getattr(provider, function)
        nodes.append(AlgorithmicNode(node_id=name, name=name,
            description=(fn.__doc__ or function).strip(), concept_type='custom', status=NodeStatus.ATOMIC,
            matched_primitive=fn.__module__+'.'+fn.__name__,
            inputs=[IOSpec(name=k, type_desc=p.annotation.__name__)
                    for k,p in inspect.signature(fn).parameters.items()],
            outputs=[IOSpec(name=k, type_desc=t) for k,t in outputs]))

    def edge(source, output, target, port, kind='object'):
        edges.append(DependencyEdge(source_id=source, output_name=output,
            target_id=target, input_name=port, source_type=kind, target_type=kind))

    node('policy', 'source_entry_policy', [(f'entry_{i}', 'int') for i in range(1,34)])
    node('prepare', 'prepare_first', [('inputs', 'object')])
    for entry in range(1,24):
        name=f'entry_{entry:02}'
        node(name, 'fit_first_entry', [('block', 'object')])
        edge('prepare','inputs',name,'inputs')
        edge('policy',f'entry_{entry}',name,'entry','int')
    for group in groups:
        representative=group[0]
        bank=f'bank_{representative}'
        node(bank,'fit_neighbor_bank',[('bank','object')])
        edge('prepare','inputs',bank,'inputs')
        edge('policy',f'entry_{representative}',bank,'representative','int')
        for entry in group:
            name=f'entry_{entry:02}'
            node(name,'select_neighbor_entry',[('block','object')])
            edge('prepare','inputs',name,'inputs')
            edge('policy',f'entry_{entry}',name,'entry','int')
            edge(bank,'bank',name,'bank')
    node('collect','collect_first',[('first_level','dict')])
    for entry in range(1,34):
        edge(f'entry_{entry:02}','block','collect',f'block_{entry}')
    return CDGExport(nodes=nodes, edges=edges, metadata={
        'publication_status':'draft','num_nodes':len(nodes),'num_edges':len(edges),
        'source_version_ids':['dc4969f7-e5a4-5644-a9d0-fa10493458ed'],
        'knn_groups':[list(g) for g in groups],
        'scope':'All 33 first-level source entries; one ten-column KNN fit per declared control group.',
        'limitations':['Supplied fixed population embedding; no new transductive or out-of-sample claim.',
                       'Source adapters retain nine classes, five folds and minimum reference counts.',
                       'Group members must have identical runtime controls; incompatible banks are rejected.',
                       'Supplemental features and meta learners are separate graphs; no complete-intake approval.']})
