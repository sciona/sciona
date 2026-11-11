"""Corrected draft execution graph for the full Contrails solution."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus


def build_contrails_graph():
    stages = [('prepare', 'payload', 'prepared', 'dict', 'object'),
              ('execute', 'prepared', 'result', 'object', 'dict')]
    nodes = [AlgorithmicNode(node_id=name, name=name, description='Contrails ' + name,
                             concept_type='custom', status=NodeStatus.ATOMIC,
                             matched_primitive='sciona.atoms.dl.contrails_execution.contrails_' + name,
                             inputs=[IOSpec(name=inp, type_desc=intype)],
                             outputs=[IOSpec(name=out, type_desc=outtype)])
             for name, inp, out, intype, outtype in stages]
    edge = DependencyEdge(source_id='prepare', target_id='execute', output_name='prepared',
                          input_name='prepared', source_type='object', target_type='object')
    return CDGExport(nodes=nodes, edges=[edge], metadata=dict(
        artifact_source='competition_execution_reconstruction', publication_status='draft',
        source_version_id='dd8cc8a7-162b-52f1-a74d-256ce661f7ba',
        source_commit='08a15beb36f9cbed4c3990e74c625c1332b61fe8',
        scope='Full two-branch MaxViT/U-Net fold training, shifted targets and augmentation, terminal checkpoints, eight-way spatial inference, unnormalized ensemble and mask encoding.',
        runtime_contract='Version1 JSON populations/config/initialization; source-sized arrays; explicit checkpoint policy; disjoint runtime keys.',
        stochastic_contract='CPU Torch, NumPy and Python seeds scoped and restored; current dependency binaries, no historical seeded equivalence claim.',
        execution_boundary='Fold training and inference remain together to release each full checkpoint after use.',
        exclusions=['Historical competition accuracy', 'Implicit pretrained downloads', 'Raw input file codecs',
                    'Tier1 human certification', 'Tier2 community-usage qualification'],
        num_nodes=2, num_edges=1))
