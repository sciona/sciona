"""Generic causal isolation-forest intake realization; no winner parity claim."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,DependencyEdge,IOSpec,NodeStatus


def build_nab_graph():
    nodes=[AlgorithmicNode(node_id=name,name=name,description='Generic NAB '+name,concept_type='custom',status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.ml.nab_execution.nab_'+name,
        inputs=[IOSpec(name=inp,type_desc=it)],outputs=[IOSpec(name=out,type_desc=ot)])
        for name,inp,out,it,ot in [('prepare','payload','prepared','dict','object'),('execute','prepared','result','object','dict')]]
    return CDGExport(nodes=nodes,edges=[DependencyEdge(source_id='prepare',target_id='execute',output_name='prepared',input_name='prepared',source_type='object',target_type='object')],metadata={
        'artifact_source':'competition_generic_execution_reconstruction','publication_status':'draft',
        'source_version_ids':['e78e1ac4-49ad-5fe2-9a58-c82862d11ec7'],
        'source_commit':'ea702d75cc2258d9d7dd35ca8e5e2539d71f3140',
        'scope':'Univariate causal window isolation forest, prior-score rolling three-sigma threshold, corrected row-index NAB scoring and null/perfect normalization.',
        'corrections':['Explicit new detector configuration; not a historical NAB detector.','Thresholds below every score are supported.','Singleton post-window penalty scale is one row.','Undefined no-window normalization is null.'],
        'exclusions':['Historical winning detector or leaderboard parity','Numenta HTM','Autoencoder alternative','Independent generalization accuracy','Elapsed-time weighting','Tier1 or Tier2 certification'],
        'num_nodes':2,'num_edges':1})
