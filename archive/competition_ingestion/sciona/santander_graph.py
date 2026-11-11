"""Draft graph for the complete independent Santander pseudo-label ensemble."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,DependencyEdge,IOSpec,NodeStatus


def build_santander_graph():
    nodes=[AlgorithmicNode(node_id=name,name=name,description=description,concept_type='custom',status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.ml.santander_execution.santander_'+name,
        inputs=[IOSpec(name=inp,type_desc=it)],outputs=[IOSpec(name=out,type_desc=ot)])
        for name,description,inp,out,it,ot in (
            ('prepare','Validate private disjoint populations and complete lifecycle controls','payload','prepared','dict','object'),
            ('execute','Fit initial neural ranks, pseudo-label both branches, regenerate features, retrain and rank blend','prepared','result','object','dict'))]
    return CDGExport(nodes=nodes,edges=[DependencyEdge(source_id='prepare',target_id='execute',output_name='prepared',input_name='prepared',source_type='object',target_type='object')],metadata={
        'artifact_source':'competition_source_corrected_execution','publication_status':'draft',
        'source_version_ids':['743c3225-058e-5a7e-ab18-ae5dd1c2b7d4'],
        'scope':'Independent complete Santander supervised-uniqueness, neural/tree augmentation, hard pseudo-label retraining and rank blend.',
        'choices':['Ten stratified folds per stage, fifteen neural epochs, explicit seeds and tree controls',
            'Full default pseudo tails: neural 5000/3000, tree 2700/2000; at least 8000 query rows',
            'Selected query rows retained in unlabeled count reference and joint labeled pseudo folds',
            'Global supervised encoding, mean substitution and transductive neural scaling',
            'Triplet-preserving augmentation and final neural/tree rank blend 2.1:1'],
        'exclusions':['Independent generalization estimates from globally encoded CV',
            'Exact historical tree settings, pseudo-reference membership or checkpoint replay',
            'Historical competitive accuracy, GPU or production deployment'],
        'num_nodes':2,'num_edges':1})
