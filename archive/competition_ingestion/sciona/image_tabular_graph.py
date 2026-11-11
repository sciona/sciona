"""Complete generic five-stage image_tabular lifecycle behind validated input and execution boundaries."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,DependencyEdge,IOSpec,NodeStatus


def build_image_tabular_graph():
    nodes=[AlgorithmicNode(node_id=name,name=name,description='Generic Image Tabular '+name,concept_type='custom',status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.ml.image_tabular_execution.image_tabular_'+name,
        inputs=[IOSpec(name=inp,type_desc=it)],outputs=[IOSpec(name=out,type_desc=ot)])
        for name,inp,out,it,ot in [('prepare','payload','prepared','dict','object'),('execute','prepared','result','object','dict')]]
    return CDGExport(nodes=nodes,edges=[DependencyEdge(source_id='prepare',target_id='execute',output_name='prepared',input_name='prepared',source_type='object',target_type='object')],metadata={
        'artifact_source':'competition_generic_execution_reconstruction','publication_status':'draft',
        'source_version_ids':['4d7b7da1-6568-56b0-aa5e-7e2e7a275556'],
        'scope':'Generic binary image/tabular fusion: handcrafted RGB descriptors, fold-fitted metadata, fused scaling, dropout label-smoothed MLPs and held-out threshold.',
        'choices':['31 fixed color/texture/aspect descriptors from normalized RGB','Fold-local median imputation, categorical encoding and fused standardization','Group-consistent folds with complete OOF coverage','CPU float64 dropout MLP and smoothed BCE; equal retained-fold probability mean','Separate group-disjoint F1 calibration with highest-threshold tie rule'],
        'exclusions':['Learned visual backbone or raw-file decoding','Historical winning recipe or empirical accuracy guarantee','Full-data model refit','Large-scale resource or cross-platform qualification'],
        'num_nodes':2,'num_edges':1})
