"""Reusable materialized model-bank handoff subgraph, with no domain constants."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus


def build_model_slot_graph():
    contracts = {
        'models': ('dict', 'Named nonmissing opaque runtime model objects; all models must be referenced. Private when learned from non-public inputs.'),
        'population_bindings': ('dict', 'Exact population names mapped to nonempty local slot-to-model-name mappings. No implicit normalization or population inference.'),
        'shared_bindings': ('dict', 'Shared slot-to-model-name mapping; slots disjoint from every local mapping. Requires populations when nonempty.'),
        'bank': ('dict', 'Owned population/slot containers retaining native model object aliases and shared identity. No model serialization or copying.'),
        'population': ('str', 'Exact nonempty declared population name; unknown names raise KeyError.'),
        'slots': ('dict', 'Owned selected slot map retaining the native model objects and aliases from the bank.'),
    }
    def port(name):
        typ, contract = contracts[name]
        return IOSpec(name=name, type_desc=typ, constraints=contract)
    nodes = []
    for node_id, function, inputs, outputs in [
        ('assemble', 'assemble_bank', ['models', 'population_bindings', 'shared_bindings'], ['bank']),
        ('select', 'select_slots', ['bank', 'population'], ['slots']),
    ]:
        nodes.append(AlgorithmicNode(node_id=node_id, name=function, description='Explicit reusable model-state handoff',
            concept_type='custom', status=NodeStatus.ATOMIC,
            matched_primitive='sciona.atoms.ml.model_selection.model_banks.'+function,
            inputs=[port(name) for name in inputs], outputs=[port(name) for name in outputs]))
    edges = [DependencyEdge(source_id='assemble', target_id='select', output_name='bank', input_name='bank',
                            source_type='dict', target_type='dict')]
    return CDGExport(nodes=nodes, edges=edges, metadata=dict(artifact_source='reusable_model_state_handoff',
        publication_status='draft', scope='Materialized bank assembly and single-population model selection.',
        exclusions=['Native model serialization', 'Model training', 'Prediction computation', 'Whole-graph static propagation']))
