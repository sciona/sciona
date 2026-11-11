"""Complete generic five-stage video multilabel lifecycle behind validated input and execution boundaries."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,DependencyEdge,IOSpec,NodeStatus


def build_video_multilabel_graph():
    nodes=[AlgorithmicNode(node_id=name,name=name,description='Generic Video Multilabel '+name,concept_type='custom',status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.ml.video_multilabel_execution.video_multilabel_'+name,
        inputs=[IOSpec(name=inp,type_desc=it)],outputs=[IOSpec(name=out,type_desc=ot)])
        for name,inp,out,it,ot in [('prepare','payload','prepared','dict','object'),('execute','prepared','result','object','dict')]]
    return CDGExport(nodes=nodes,edges=[DependencyEdge(source_id='prepare',target_id='execute',output_name='prepared',input_name='prepared',source_type='object',target_type='object')],metadata={
        'artifact_source':'competition_generic_execution_reconstruction','publication_status':'draft',
        'source_version_ids':['13b48adf-8fb5-5ea2-a26c-44344bb8907e'],
        'scope':'Generic precomputed-feature video multilabel lifecycle: batch-padded frame loading, learned attention pooling, context-gated head, sparse logistic fusion and per-label F1 thresholds.',
        'choices':['Variable-length frame matrices and nonnegative sparse mappings','Masked attention without temporal position encoding','CPU float64 Adam/BCE neural training and per-label sparse logistic models','Equal neural/sparse probability mean','Disjoint training/calibration/query groups; highest-threshold F1 tie rule'],
        'exclusions':['Raw-video feature extraction or historical winning recipe','Frame-order sensitivity','Rare-label accuracy guarantee','Disk streaming, large-scale resource or cross-platform qualification'],
        'num_nodes':2,'num_edges':1})
