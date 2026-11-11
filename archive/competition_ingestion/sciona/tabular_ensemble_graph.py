"""Complete generic five-stage tabular lifecycle behind validated input and execution boundaries."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,DependencyEdge,IOSpec,NodeStatus


def build_tabular_ensemble_graph():
    nodes=[AlgorithmicNode(node_id=name,name=name,description='Generic Tabular Ensemble '+name,concept_type='custom',status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.ml.tabular_ensemble_execution.tabular_ensemble_'+name,
        inputs=[IOSpec(name=inp,type_desc=it)],outputs=[IOSpec(name=out,type_desc=ot)])
        for name,inp,out,it,ot in [('prepare','payload','prepared','dict','object'),('execute','prepared','result','object','dict')]]
    return CDGExport(nodes=nodes,edges=[DependencyEdge(source_id='prepare',target_id='execute',output_name='prepared',input_name='prepared',source_type='object',target_type='object')],metadata={
        'artifact_source':'competition_generic_execution_reconstruction','publication_status':'draft',
        'source_version_ids':['9f3c8c4f-ce85-5e9d-b169-8b055562ce65'],
        'scope':'Generic binary mixed-table cleaning, fold-local features, logistic/ExtraTrees model zoo, OOF logistic stacking, held-out sigmoid calibration and query prediction.',
        'choices':['Caller-supplied contiguous group-consistent folds','Fold-local medians, quantile clipping, numeric constant/duplicate removal, one-hot and frequency features','Two base models per fold and full-training refit','Disjoint training/calibration/query groups','Regularized sigmoid calibration and fixed 0.5 decision'],
        'exclusions':['Historical winning solution or accuracy','Empirical calibration guarantee','Automatic group discovery or chronological validation','Large dense categorical resource qualification'],
        'num_nodes':2,'num_edges':1})
