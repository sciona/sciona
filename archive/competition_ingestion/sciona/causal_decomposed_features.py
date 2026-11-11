"""Explicit 43-column training/prediction feature construction and five-model graph."""
from sciona.architect.models import AlgorithmicNode,DependencyEdge,IOSpec,NodeStatus
from sciona.atoms.causal_inference.feature_primitives.selected_features import FEATURE_ORDER
from sciona.causal_decomposed_prediction import build_explicit_causal_prediction_graph


def build_explicit_causal_feature_graph():
    graph=build_explicit_causal_prediction_graph()
    graph.nodes=[n for n in graph.nodes if n.node_id!='features']
    downstream=[e for e in graph.edges if e.source_id=='features']
    graph.edges=[e for e in graph.edges if e.source_id!='features']
    prefix='sciona.atoms.causal_inference.feature_primitives.'
    array='numpy.ndarray'
    def node(identity,primitive,inputs,outputs,description):
        graph.nodes.append(AlgorithmicNode(node_id=identity,name=identity.replace('_',' '),
            description=description,concept_type='custom',status=NodeStatus.ATOMIC,
            matched_primitive=prefix+primitive,
            inputs=[IOSpec(name=n,type_desc=t) for n,t in inputs],
            outputs=[IOSpec(name=n,type_desc=t) for n,t in outputs]))
    def edge(source,target,output,input_name,kind):
        graph.edges.append(DependencyEdge(source_id=source,target_id=target,output_name=output,
            input_name=input_name,source_type=kind,target_type=kind))
    ports=[(n,'list') for n in ['training_pairs','prediction_pairs','training_types','prediction_types']]
    node('feature_inputs','feature_recipe_ports.route_feature_populations',ports,ports,
        'Route independent populations without computing statistics or using labels.')
    node('feature_configuration','feature_recipe_ports.feature_statistic_configuration',[],
        [(n,'str') for n in FEATURE_ORDER],'Declare fixed recipe statistic choices explicitly.')
    for population in ['training','prediction']:
        orientation=population+'_orientation';assembly=population+'_matrix'
        node(orientation,'pair_orientation.orient_typed_pairs',[('pairs','list'),('variable_types','list')],
            [(n,'list') for n in ['xs','x_types','ys','y_types']],
            'Validate typed pairs and emit adjacent forward/reverse observations.')
        edge('feature_inputs',orientation,population+'_pairs','pairs','list')
        edge('feature_inputs',orientation,population+'_types','variable_types','list')
        node(assembly,'feature_recipe_ports.assemble_selected_columns',[(n,array) for n in FEATURE_ORDER],
            [('matrix',array)],'Assemble the fixed source column order from explicitly computed statistics.')
        for feature in FEATURE_ORDER:
            identity=population+'_'+feature
            node(identity,'batched_statistics.typed_pair_statistic',
                [(n,'list') for n in ['xs','x_types','ys','y_types']]+[('statistic','str')],
                [('column',array)],'Compute one declared statistic independently for each typed observation pair.')
            reverse=feature.endswith(('_ba','_b'))
            for target,source in [('xs','ys' if reverse else 'xs'),('x_types','y_types' if reverse else 'x_types'),
                                  ('ys','xs' if reverse else 'ys'),('y_types','x_types' if reverse else 'y_types')]:
                edge(orientation,identity,source,target,'list')
            edge('feature_configuration',identity,feature,'statistic','str')
            edge(identity,assembly,'column',feature,array)
    node('augment_labels','pair_orientation.augment_pair_labels',[('pair_labels',array),('pairs','list')],
        [('labels',array)],'Interleave signed forward/reverse labels without reading prediction observations.')
    edge('feature_inputs','augment_labels','training_pairs','pairs','list')
    replacements={'training_features':('training_matrix','matrix'),
                  'training_labels':('augment_labels','labels'),'X':('prediction_matrix','matrix')}
    for old in downstream:
        source,output=replacements[old.output_name]
        edge(source,old.target_id,output,old.input_name,array)
    graph.metadata.update(scope='Explicit typed-pair orientation, 43 statistic columns per population, ordered assembly, five fits and five predictions.',
        num_nodes=len(graph.nodes),num_edges=len(graph.edges),publication_status='draft')
    return graph
