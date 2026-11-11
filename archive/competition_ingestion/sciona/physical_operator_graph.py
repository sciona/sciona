"""Complete generic five-stage physical operator lifecycle behind validated input and execution boundaries."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,DependencyEdge,IOSpec,NodeStatus


def build_physical_operator_graph():
    nodes=[AlgorithmicNode(node_id=name,name=name,description='Generic Physical Operator '+name,concept_type='custom',status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.ml.physical_operator_execution.physical_operator_'+name,
        inputs=[IOSpec(name=inp,type_desc=it)],outputs=[IOSpec(name=out,type_desc=ot)])
        for name,inp,out,it,ot in [('prepare','payload','prepared','dict','object'),('execute','prepared','result','object','dict')]]
    return CDGExport(nodes=nodes,edges=[DependencyEdge(source_id='prepare',target_id='execute',output_name='prepared',input_name='prepared',source_type='object',target_type='object')],metadata={
        'artifact_source':'competition_generic_execution_reconstruction','publication_status':'draft',
        'source_version_ids':['49c645b7-3659-5cfa-a386-78f7a51f7358'],
        'scope':'Generic periodic scalar diffusion: SI state assembly, Fourier features, learned spectral/local surrogate, nonnegative fixed-mass projection and conservative periodic smoothing.',
        'choices':['Uniform endpoint-excluded grid and nondimensional diffusivity*time/length^2','CPU float64 full-batch Adam Fourier operator','Group-disjoint training/validation/query populations','Earliest best validation checkpoint including epoch zero','Euclidean nonnegative mass-simplex projection and nearest-neighbor convex smoothing'],
        'exclusions':['Historical winning solution or original FNO benchmark reproduction','Exact PDE solver or arbitrary PDE accuracy','Nonuniform or nonperiodic grids','Empirical uncertainty estimates and deployment resource qualification'],
        'num_nodes':2,'num_edges':1})
