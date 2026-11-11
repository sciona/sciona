"""Generic aggregate/lag/encode/LightGBM search and one-period forecast graph."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,DependencyEdge,IOSpec,NodeStatus


def build_future_sales_graph():
    nodes=[AlgorithmicNode(node_id=name,name=name,description='Generic Future Sales '+name,concept_type='custom',status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.ml.future_sales_execution.future_sales_'+name,
        inputs=[IOSpec(name=inp,type_desc=it)],outputs=[IOSpec(name=out,type_desc=ot)])
        for name,inp,out,it,ot in [('prepare','payload','prepared','dict','object'),('execute','prepared','result','object','dict')]]
    return CDGExport(nodes=nodes,edges=[DependencyEdge(source_id='prepare',target_id='execute',output_name='prepared',input_name='prepared',source_type='object',target_type='object')],metadata={
        'artifact_source':'competition_generic_execution_reconstruction','publication_status':'draft',
        'source_version_ids':['e6803752-90ef-5cab-8b30-5ebeec716c15'],
        'scope':'Explicit period aggregation then clipping, prior-only lag/rolling/group means, bounded TPE LightGBM search with early stopping and selected-round final refit.',
        'choices':['Dedicated same-environment worker process for native training','Fixed caller entity universe and zero absence semantics','Rolling one-step validation with prior-period observations available','Single final forecast period; bounded predictions'],
        'exclusions':['Historical winning recipe or accuracy','Fixed-origin multiple-period forecasting','XGBoost or optional ensemble branch','Holiday or promotion inputs and learned embeddings','Unbiased hyperparameter-selection performance','Tier1 or Tier2 certification'],
        'num_nodes':2,'num_edges':1})
