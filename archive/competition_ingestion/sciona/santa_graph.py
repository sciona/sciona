"""Corrected Santa lifecycle with private replay and visible-history inputs."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus


def build_santa_graph():
    nodes = [AlgorithmicNode(node_id=name, name=name, description='Corrected Santa ' + name,
             concept_type='custom', status=NodeStatus.ATOMIC,
             matched_primitive='sciona.atoms.ml.santa_execution.santa_' + name,
             inputs=[IOSpec(name=inp, type_desc=it)], outputs=[IOSpec(name=out, type_desc=ot)])
             for name, inp, out, it, ot in [('prepare', 'payload', 'prepared', 'dict', 'object'),
                                         ('execute', 'prepared', 'result', 'object', 'dict')]]
    return CDGExport(nodes=nodes, edges=[DependencyEdge(source_id='prepare', target_id='execute',
        output_name='prepared', input_name='prepared', source_type='object', target_type='object')], metadata={
        'artifact_source':'competition_source_corrected_execution','publication_status':'draft',
        'source_version_ids':['8ab2dc68-65da-5415-bf3a-bbebde6fe898'],
        'scope':'Corrected Santa winner method: discrete threshold beliefs, opponent features, two LightGBM regressors, decayed sigmoid blend and action selection.',
        'choices':['Discrete initial-threshold posterior from own rewards under shared decay',
                   'Ten pre-round features including opponent action statistics',
                   'Disjoint replay games and seeded one-in-five row sampling',
                   'Raw and exponential target RMSE LightGBM models with held-out early stopping',
                   'Normalized sigmoid exploration blend, shared decay and seeded randomized ties'],
        'exclusions':['Original intake Beta/UCB description is contradicted by winner source',
                      'Historical training replay membership, weights or winning performance',
                      'Opponent rewards or hidden thresholds at inference',
                      'Resource or cross-platform qualification'],
        'num_nodes':2,'num_edges':1})
