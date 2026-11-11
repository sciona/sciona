"""Complete generic five-stage geotemporal lifecycle behind validated input and execution boundaries."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,DependencyEdge,IOSpec,NodeStatus


def build_geotemporal_graph():
    nodes=[AlgorithmicNode(node_id=name,name=name,description='Generic Geotemporal '+name,concept_type='custom',status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.ml.geotemporal_execution.geotemporal_'+name,
        inputs=[IOSpec(name=inp,type_desc=it)],outputs=[IOSpec(name=out,type_desc=ot)])
        for name,inp,out,it,ot in [('prepare','payload','prepared','dict','object'),('execute','prepared','result','object','dict')]]
    return CDGExport(nodes=nodes,edges=[DependencyEdge(source_id='prepare',target_id='execute',output_name='prepared',input_name='prepared',source_type='object',target_type='object')],metadata={
        'artifact_source':'competition_generic_execution_reconstruction','publication_status':'draft',
        'source_version_ids':['9bb6428c-0529-56c8-b51d-96e1b4d6990c'],
        'scope':'Generic causal planar geotemporal regression: available context joins, spatial/lag features, forward validation, ExtraTrees fusion and nonnegative same-time smoothing.',
        'choices':['Shared planar metric reference and seconds clock','Latest available context per site and strictly historical target lags','Expanding time blocks with explicit gap and unvalidated warmup','Fold-local history and final full-training refit','Same-time query neighbor smoothing after nonnegative clipping'],
        'exclusions':['Automatic coordinate transformation or clock alignment','Unseen-location accuracy or historical winning recipe','OOF predictions for fitting warmup','Query-population independence after geographic smoothing','Large-scale resource qualification'],
        'num_nodes':2,'num_edges':1})
