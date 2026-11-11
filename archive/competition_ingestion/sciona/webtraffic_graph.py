"""Derived Community Web Traffic execution graph; publication review required."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,DependencyEdge,IOSpec,NodeStatus


def build_webtraffic_graph():
    stages=[('prepare','webtraffic_prepare','payload','prepared'),('train','webtraffic_train','prepared','completed'),
            ('forecast','webtraffic_forecast','completed','result')]
    nodes=[AlgorithmicNode(node_id=node,name=node,description=primitive,concept_type='custom',status=NodeStatus.ATOMIC,
                           matched_primitive='sciona.atoms.dl.webtraffic_execution.'+primitive,
                           inputs=[IOSpec(name=inp,type_desc='dict' if node=='prepare' else 'object')],
                           outputs=[IOSpec(name=out,type_desc='dict' if node=='forecast' else 'object')])
           for node,primitive,inp,out in stages]
    edges=[DependencyEdge(source_id='prepare',target_id='train',output_name='prepared',input_name='prepared',source_type='object',target_type='object'),
           DependencyEdge(source_id='train',target_id='forecast',output_name='completed',input_name='completed',source_type='object',target_type='object')]
    return CDGExport(nodes=nodes,edges=edges,metadata=dict(artifact_source='competition_execution_reconstruction',publication_status='draft',
        source_version_id='a6298d16-4fa7-5577-b559-7c17c006ccd5',source_commit='a9abb80c800409abf0ece21ea244ef779f758f96',
        scope='Runtime daily series through s32 three-GRU training, Adam, declared EMA checkpoints and dated63-day ensemble forecasts.',
        runtime_contract='Version1 payload, sorted unique page strings, nonnegative finite counts/null, contiguous daily range and explicit RuntimeConfig.',
        stochastic_contract='Independent per-model NumPy permutation/offset and CPU torch initialization/dropout streams; not original TensorFlow seeded replay.',
        ema_contract='Explicit before/after parameter and shared-step observation phases; default after/after, not source race reproduction.',
        exclusions=['unconsumed s32 attention branch','original TensorFlow checkpoint file codec','competition-specific submission/date selection',
                    'historical competition accuracy','Tier1 certification','Tier2 community usage qualification'],num_nodes=3,num_edges=2))
