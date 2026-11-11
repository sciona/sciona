"""Draft full adversarial runtime graph; lower-tier publication requires review gates."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus


def build_adversarial_graph():
    stages=[('prepare','payload','prepared','dict','object'),
            ('execute','prepared','result','object','dict')]
    nodes=[AlgorithmicNode(node_id=name,name=name,description='adversarial '+name,
                          concept_type='custom',status=NodeStatus.ATOMIC,
                          matched_primitive='sciona.atoms.dl.adversarial_execution.adversarial_'+name,
                          inputs=[IOSpec(name=inp,type_desc=intype)],
                          outputs=[IOSpec(name=out,type_desc=outtype)])
           for name,inp,out,intype,outtype in stages]
    return CDGExport(nodes=nodes,edges=[DependencyEdge(source_id='prepare',target_id='execute',
                     output_name='prepared',input_name='prepared',source_type='object',target_type='object')],
                     metadata=dict(artifact_source='competition_execution_reconstruction',publication_status='draft',
                     source_version_id='cebbfc7d-e32e-5f52-9ec2-e1de0d955501',
                     source_commits=['5c68162e05de7afed2e3d33115df43e2a7a1c3da','7da707fe568053a9e2735fcc692c8f0ad7e122a1'],
                     scope='Full8/5/2-model main/aux objectives, frozen labels, source momentum and10/20/40iterations, ten-entry padded RGB batches and real-entry PNG output.',
                     runtime_contract='Version1 JSON arrays and explicit random or caller-decoded Slim model states. No implicit model downloads.',
                     numerical_contract='CPU Torch adaptation; historical TensorFlow binary, checkpoint identity and max-pool tie-gradient parity unverified. Undefined gradient normalization rejects.',
                     output_contract='Normalized epsilon projection applies before source SciPy minmax PNG scaling; saved PNG pixel bound is not guaranteed.',
                     exclusions=['Historical competition accuracy','Implicit pretrained downloads','TensorFlow checkpoint binary decoding',
                                 'Tier1 human certification','Tier2 usage qualification'],
                     num_nodes=2,num_edges=1))
