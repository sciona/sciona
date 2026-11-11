"""Explicit frozen preprocessing, conversion, fold replay and reduction graph."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,DependencyEdge,IOSpec,NodeStatus


def build_quantile_replay_graph():
    def node(identity,fqdn,inputs,outputs,description):
        return AlgorithmicNode(node_id=identity,name=identity,description=description,
            concept_type='custom',status=NodeStatus.ATOMIC,matched_primitive=fqdn,
            inputs=[IOSpec(name=n,type_desc=t,constraints='Explicit caller-owned numeric state, mapped query scope and coordinates; incoming edges preserve declared order.') for n,t in inputs],
            outputs=[IOSpec(name=n,type_desc=t,constraints='Preserve qualified provider semantics, order and caller coordinate meanings.') for n,t in outputs])
    domain='sciona.atoms.ml.domain_adapters.'
    typed='sciona.atoms.ml.pipeline.typed_inputs.'
    coordinates=[('feature_coordinate','str'),('response_coordinate','str')]
    nodes=[
        node('features',domain+'osic_feature_replay.transform_frozen_queries',
             [('state','dict'),('queries','pandas.DataFrame')]+coordinates,[('result','dict')],
             'Transform the exact declared query batch with frozen private state; no fitting.'),
        node('project',domain+'osic_feature_projection.unpack_quantile_features',
             [('prepared','dict')],[('features','numpy.ndarray'),('columns','list'),('mask','numpy.ndarray'),('values','numpy.ndarray')],
             'Expose borrowed feature outputs as separate named ports without conversion.'),
        node('cast',typed+'cast_finite_array',[('values','numpy.ndarray'),('dtype','str')],[('result','numpy.ndarray')],
             'Explicit source float32 model-input conversion; caller supplies dtype=float32.'),
        node('columns',typed+'encode_feature_descriptors',[('descriptors','list')],[('result','list[str]')],
             'Encode structured column descriptors as exact compact JSON strings.'),
        node('replay',domain+'osic_quantile.replay_prepared_quantile_folds',
             [('states','list'),('features','numpy.ndarray'),('columns','list[str]')]+coordinates,[('result','numpy.ndarray')],
             'Restore two private numeric fold states and produce raw fold predictions; no fitting.'),
        node('reduce',domain+'osic_quantile.reduce_quantile_predictions',
             [('folds','numpy.ndarray'),('observed_mask','numpy.ndarray'),('observed_values','numpy.ndarray')],[('result','numpy.ndarray')],
             'Apply source fold averaging, raw spread and explicitly available observation policy.')]
    links=[('features','project','result','prepared','dict'),
           ('project','cast','features','values','numpy.ndarray'),
           ('project','columns','columns','descriptors','list'),
           ('cast','replay','result','features','numpy.ndarray'),
           ('columns','replay','result','columns','list[str]'),
           ('replay','reduce','result','folds','numpy.ndarray'),
           ('project','reduce','mask','observed_mask','numpy.ndarray'),
           ('project','reduce','values','observed_values','numpy.ndarray')]
    edges=[DependencyEdge(source_id=s,target_id=t,output_name=o,input_name=i,source_type=k,target_type=k)
           for s,t,o,i,k in links]
    return CDGExport(nodes=nodes,edges=edges,metadata=dict(num_nodes=6,num_edges=8,
        scope='Frozen mapped-query quantile inference with explicit preparation, conversion, replay and reduction.',
        required_source_policy=dict(dtype='float32'),
        exclusions=['Raw query-key extraction and final branch blending','Complete image training and original CDG approval',
                    'HTTP transport and empirical effectiveness']))
