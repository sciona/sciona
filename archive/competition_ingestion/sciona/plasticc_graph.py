"""Derived Community Avocado execution graph; publication review required."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,DependencyEdge,IOSpec,NodeStatus


def build_plasticc_graph():
    stages=[('prepare','plasticc_prepare','payload','prepared'),('train','plasticc_train','prepared','completed'),
            ('predict','plasticc_predict','completed','result')]
    nodes=[AlgorithmicNode(node_id=node,name=node,description=primitive,concept_type='custom',status=NodeStatus.ATOMIC,
                           matched_primitive='sciona.atoms.dl.plasticc_execution.'+primitive,
                           inputs=[IOSpec(name=inp,type_desc='dict' if node=='prepare' else 'object')],
                           outputs=[IOSpec(name=out,type_desc='dict' if node=='predict' else 'object')])
           for node,primitive,inp,out in stages]
    edges=[DependencyEdge(source_id='prepare',target_id='train',output_name='prepared',input_name='prepared',source_type='object',target_type='object'),
           DependencyEdge(source_id='train',target_id='predict',output_name='completed',input_name='completed',source_type='object',target_type='object')]
    return CDGExport(nodes=nodes,edges=edges,metadata=dict(
        artifact_source='competition_execution_reconstruction', publication_status='draft',
        source_version_id='57909e28-f500-51c8-91da-8e104377d2fd',
        source_commit='cd7809db4d7eb92860b178d8f5a53ab16f03ad91',
        scope='Explicit observations and empirical reference through augmentation, 2D Matern GP, custom features, grouped LightGBM folds and task-specific probability adjustment.',
        runtime_contract='Version1 training/prediction/reference/config envelope; disjoint populations; source class closure; finite numerical observations.',
        stochastic_contract='Instance NumPy RandomState augmentation, explicit fold and model seeds; current LightGBM callbacks, no historical binary parity claim.',
        exclusions=['File persistence and submission CSV codecs','optional bias-simulation experiment',
            'historical competition accuracy','general novelty detection','Tier1 certification','Tier2 community usage qualification'],
        num_nodes=3,num_edges=2))
