"""Decomposed raw-feature CDG reusing approved generic feature operations."""
import sciona.atoms.ml.tabular.available_features as features
from sciona.available_feature_contracts import contracts
from sciona.nasa_first_graph_adapters import OPERATIONS
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus


def build_nasa_first_feature_graph():
    nodes, edges = [], []
    generic = 'sciona.atoms.ml.tabular.available_features.'
    adapter = 'sciona.atoms.ml.domain_adapters.first_place_features.'
    context = IOSpec(name='context', type_desc='dict', constraints='Private runtime row identities, grid and explicit naming policies for output projection; no learned parameters.')
    prepared = [context]
    projected = [context]
    def node(identity, runtime, inputs, outputs):
        return AlgorithmicNode(node_id=identity, name=runtime.rsplit('.', 1)[-1], description='Decomposed first-place raw-feature operation',
            concept_type='custom', status=NodeStatus.ATOMIC, matched_primitive=runtime, inputs=inputs, outputs=outputs)
    def edge(source, target, output, input_port):
        edges.append(DependencyEdge(source_id=source, target_id=target, output_name=output,
            input_name=input_port.name, source_type=input_port.type_desc, target_type=input_port.type_desc))
    edge('prepare', 'project', 'context', context)
    for identity, symbol in OPERATIONS:
        inputs, outputs = contracts(symbol, getattr(features, symbol))
        inputs, outputs = [IOSpec(**port) for port in inputs], [IOSpec(**port) for port in outputs]
        nodes.append(node(identity, generic+symbol, inputs, outputs))
        for port in inputs:
            name = identity+'_'+port.name
            prepared.append(port.model_copy(update={'name': name}))
            edge('prepare', identity, name, port)
        for port in outputs:
            name = identity+'_'+port.name if len(outputs)>1 else identity
            mapped = port.model_copy(update={'name': name})
            projected.append(mapped)
            edge(identity, 'project', port.name, mapped)
    join_inputs, join_outputs = contracts('assemble_tables', features.assemble_tables)
    join_inputs, join_outputs = [IOSpec(**p) for p in join_inputs], [IOSpec(**p) for p in join_outputs]
    nodes.insert(0, node('prepare', adapter+'prepare', [
        IOSpec(name='queries', type_desc='pd.DataFrame', constraints='Caller-owned public-software query fields on one naive clock; duplicate rows and input index retained.'),
        IOSpec(name='raw_options', type_desc='dict', constraints='Explicit raw tables, context interval, vocabularies, holiday schedule and entity snapshot availability; never inferred from private files.')], prepared))
    nodes.append(node('project', adapter+'project', projected, join_inputs))
    nodes.append(node('assemble', generic+'assemble_tables', join_inputs, join_outputs))
    for port in join_inputs:
        edge('project', 'assemble', port.name, port)
    return CDGExport(nodes=nodes, edges=edges, metadata=dict(artifact_source='decomposed_first_place_features',
        publication_status='draft', scope='All seven raw feature families and identity-preserving assembly through nine reusable operations.',
        exclusions=['Model fitting', 'Final model imputation', 'Prediction and fallback routing', 'Static symbolic propagation']))
