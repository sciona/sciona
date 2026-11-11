"""TrackML full iterative lifecycle graph; publication review required."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,DependencyEdge,IOSpec,NodeStatus


def build_trackml_graph():
    prefix='sciona.atoms.physics.trackml_execution.'
    stages=[('decode','trackml_decode','payload','prepared'),
            ('track','trackml_track','prepared','assignments'),
            ('encode','trackml_encode','assignments','result')]
    nodes=[AlgorithmicNode(node_id=identity,name=identity,description=primitive,
        concept_type='custom',status=NodeStatus.ATOMIC,matched_primitive=prefix+primitive,
        inputs=[IOSpec(name=input_name,type_desc='dict' if identity=='decode' else 'object')],
        outputs=[IOSpec(name=output_name,type_desc='dict' if identity=='encode' else 'object')])
        for identity,primitive,input_name,output_name in stages]
    edges=[DependencyEdge(source_id='decode',target_id='track',output_name='prepared',input_name='prepared',source_type='object',target_type='object'),
           DependencyEdge(source_id='track',target_id='encode',output_name='assignments',input_name='assignments',source_type='object',target_type='object')]
    return CDGExport(nodes=nodes,edges=edges,metadata=dict(
        artifact_source='competition_execution_reconstruction',publication_status='draft',
        source_version_id='11c41ece-c648-50b4-b211-7e319ba36dea',
        source_commit='f1a6e63969167159ef72c4ba32897e5aef7c3888',
        scope='Explicit runtime module/hit/cell tables and optional grid maps through complete iterative tracking and per-hit assignments.',
        runtime_contract='Every detector layer retains sufficient hits for configured kNN in every commit round; sparse/depleted populations reject.',
        lifecycle=['seed','fit','follow','pair','rank','commit','optional odd-hit postprocess','fill'],
        compatibility=['modern pandas column/Series conversion','isolated neighbor defaults','vector pre-move dispatch',
                       'empty follow population stops','no-seed completion'],
        exclusions=['learned map calibration','historical competition accuracy','truth scoring','diagnostic/file output','sparse-layer tracking'],
        num_nodes=3,num_edges=2))
