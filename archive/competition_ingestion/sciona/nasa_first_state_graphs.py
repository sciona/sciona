"""Portable policy-bound state connects complete training and inference graphs."""
from sciona.architect.models import AlgorithmicNode,DependencyEdge,IOSpec,NodeStatus
from sciona.nasa_first_raw_training_graph import build_nasa_first_raw_training_graph
from sciona.nasa_first_guarded_graph import build_nasa_first_guarded_graph


def _node(identity,runtime,inputs,outputs):
    def port(name,typ):return IOSpec(name=name,type_desc=typ,constraints='Private runtime model/policy state; preserve integrity, schema and training/inference policy agreement.')
    return AlgorithmicNode(node_id=identity,name=identity,description='Explicit model/policy state handoff',concept_type='data_assembly',
        status=NodeStatus.ATOMIC,matched_primitive=runtime,inputs=[port(*p) for p in inputs],outputs=[port(*p) for p in outputs])


def _edge(graph,source,output,target,name):
    before=next(port for node in graph.nodes if node.node_id==source for port in node.outputs if port.name==output)
    after=next(port for node in graph.nodes if node.node_id==target for port in node.inputs if port.name==name)
    graph.edges.append(DependencyEdge(source_id=source,target_id=target,output_name=output,input_name=name,
                                     source_type=before.type_desc,target_type=after.type_desc))


def build_nasa_first_training_state_graph():
    graph=build_nasa_first_raw_training_graph()
    graph.nodes.extend([
        _node('encode_bank','sciona.atoms.ml.model_selection.native_model_state.pack_bank',[('bank','dict')],[('model_state','dict')]),
        _node('capture_policy','sciona.atoms.ml.domain_adapters.first_place_policy.capture',[('bank','dict'),('raw_populations','dict')],[('policy','dict')]),
        _node('bind_state','sciona.atoms.ml.model_selection.policy_state.bind',[('model_state','dict'),('policy','dict')],[('state','dict')])])
    _edge(graph,'bank','bank','encode_bank','bank');_edge(graph,'bank','bank','capture_policy','bank')
    _edge(graph,'encode_bank','model_state','bind_state','model_state');_edge(graph,'capture_policy','policy','bind_state','policy')
    graph.metadata.update(scope='Complete raw training and portable model-bank output bound to training-time feature policies.',
                          exclusions=['Empirical predictive quality','Static symbolic propagation'])
    return graph


def build_nasa_first_state_inference_graph():
    graph=build_nasa_first_guarded_graph()
    graph.nodes[:0]=[
        _node('unbind_state','sciona.atoms.ml.model_selection.policy_state.unbind',[('state','dict')],[('model_state','dict'),('policy','dict')]),
        _node('decode_bank','sciona.atoms.ml.model_selection.native_model_state.unpack_bank',[('state','dict')],[('bank','dict')]),
        _node('bound_inputs','sciona.atoms.ml.domain_adapters.first_place_policy.prediction_inputs',
              [('bank','dict'),('policy','dict'),('population','str'),('raw_options','dict')],
              [('models','dict'),('raw_options','dict'),('feature_columns','list'),('numeric_fills','dict'),('categorical_fills','dict')])]
    _edge(graph,'unbind_state','model_state','decode_bank','state');_edge(graph,'decode_bank','bank','bound_inputs','bank')
    _edge(graph,'unbind_state','policy','bound_inputs','policy')
    for name in ['models','raw_options','feature_columns','numeric_fills','categorical_fills']:
        _edge(graph,'bound_inputs',name,'inputs',name)
    graph.metadata.update(scope='Portable native model restoration, bound policy validation and complete guarded inference.',
        exclusions=['Training execution in the inference call','Empirical predictive quality','Static symbolic propagation'])
    return graph
