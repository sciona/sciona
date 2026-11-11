"""Complete generic six-stage text-pair lifecycle behind validated input and execution boundaries."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,DependencyEdge,IOSpec,NodeStatus


def build_text_pair_graph():
    nodes=[AlgorithmicNode(node_id=name,name=name,description='Generic Text Pair '+name,concept_type='custom',status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.ml.text_pair_execution.text_pair_'+name,
        inputs=[IOSpec(name=inp,type_desc=it)],outputs=[IOSpec(name=out,type_desc=ot)])
        for name,inp,out,it,ot in [('prepare','payload','prepared','dict','object'),('execute','prepared','result','object','dict')]]
    return CDGExport(nodes=nodes,edges=[DependencyEdge(source_id='prepare',target_id='execute',output_name='prepared',input_name='prepared',source_type='object',target_type='object')],metadata={
        'artifact_source':'competition_generic_execution_reconstruction','publication_status':'draft',
        'source_version_ids':['1169253d-76d3-5b76-885f-07b58bed795c'],
        'scope':'Generic binary text-pair matching: normalization, lexical similarities, training-only LSA and candidate graph context, logistic/random-forest blend, held-out F1 threshold.',
        'choices':['Training-only features and base models','Current training pair excluded from graph context','Separate calibration and prediction pair populations','Equal two-model score mean and highest-threshold F1 tie rule'],
        'exclusions':['Historical winning solution or accuracy','Unseen-entity generalization','Probability calibration guarantee','External pretrained language corpus'],
        'num_nodes':2,'num_edges':1})
