"""Draft graph for the corrected independent uncertainty reconstruction."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,DependencyEdge,IOSpec,NodeStatus


def build_graph():
    nodes=[AlgorithmicNode(node_id=name,name=name,description=description,concept_type='custom',status=NodeStatus.ATOMIC,
                           matched_primitive='sciona.atoms.ml.m5u_execution.m5u_'+name,
                           inputs=[IOSpec(name=inp,type_desc=it)],outputs=[IOSpec(name=out,type_desc=ot)])
           for name,description,inp,out,it,ot in (
               ('prepare','Validate canonical private history, hierarchy, calendar and training controls','payload','prepared','dict','object'),
               ('execute','Train 14 direct-quantile levels, average repeated forecasts and restore hierarchy units','prepared','result','object','dict'))]
    return CDGExport(nodes=nodes,edges=[DependencyEdge(source_id='prepare',target_id='execute',output_name='prepared',input_name='prepared',source_type='object',target_type='object')],metadata={
        'artifact_source':'competition_source_corrected_execution','publication_status':'draft',
        'source_version_ids':['5d2f953e-86bb-57f8-9a97-e0b6a303fd94'],
        'scope':'Independent corrected direct LightGBM quantile reconstruction, all 14 configured levels and nine quantiles, 28-day forecasts.',
        'choices':['Full default per-level sampling, scale perturbation and grouped randomized search counts',
                   'One bag, latest outer holdout, global-controller quantile masks and independent deterministic CPU seeds',
                   'Full repeated query budget, inverse scaling, fold/bag blending and level median-total adjustment',
                   'Explicit retained-base-reference correction for source-undefined aggregate outlet normalization; defined source behavior preserved',
                   'Canonical private JSON input up to 256 MiB; unrounded native-unit output'],
        'exclusions':['Unsupported intake point-residual and fitted-distribution recipe',
                      'Historical code/RNG/weight parity, historical score replication, competitive accuracy and production qualification'],
        'num_nodes':2,'num_edges':1})
