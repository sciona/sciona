"""Draft full DFDC runtime graph; lower-tier publication requires review gates."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus


def build_dfdc_graph():
    stages=[('prepare','payload','prepared','dict','object'),
            ('execute','prepared','result','object','dict')]
    nodes=[AlgorithmicNode(node_id=name,name=name,description='DFDC '+name,
                          concept_type='custom',status=NodeStatus.ATOMIC,
                          matched_primitive='sciona.atoms.dl.dfdc_execution.dfdc_'+name,
                          inputs=[IOSpec(name=inp,type_desc=intype)],
                          outputs=[IOSpec(name=out,type_desc=outtype)])
           for name,inp,out,intype,outtype in stages]
    return CDGExport(nodes=nodes,edges=[DependencyEdge(source_id='prepare',target_id='execute',
                     output_name='prepared',input_name='prepared',source_type='object',target_type='object')],
                     metadata=dict(artifact_source='competition_execution_reconstruction',publication_status='draft',
                     source_version_id='95e3414b-c0ed-5816-b944-4466d697d9fc',
                     source_commit='89c6290490bac96b29193a4061b3db9dd3933e36',
                     scope='MTCNN paired crops and annotations; part folds; augmented full B7 training; explicit numbered checkpoint selection; confidence-based ensemble inference.',
                     runtime_contract='Version1 JSON positional decoded RGB clips and states; explicit native hull predictor and executable run/request plan.',
                     checkpoint_contract='Source default40 epochs cannot produce requested suffix40. No implicit extension/relabeling; all plan changes explicit and historical checkpoint identity unverified.',
                     stochastic_contract='Historical timm stochastic depth restored; current CPU binaries; caller Python/NumPy/Torch RNG restored.',
                     execution_boundary='Preparation, full training and selected-state inference stay connected; numerical tensors remain runtime-only.',
                     exclusions=['Historical competition accuracy','Implicit pretrained downloads','Media codecs',
                                 'Historical CUDA/AMP equivalence','Tier1 human certification','Tier2 usage qualification'],
                     num_nodes=2,num_edges=1))
