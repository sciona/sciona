"""Five-stage constrained-route realization of the generic planning topology."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus


def build_resource_route_graph():
    stages=[('encode', 'payload', 'state', 'dict', 'object'), ('generate', 'state', 'candidates', 'object', 'object'), ('search', 'candidates', 'searched', 'object', 'object'), ('validate', 'searched', 'validated', 'object', 'object'), ('select', 'validated', 'result', 'object', 'dict')]
    nodes=[AlgorithmicNode(node_id=name,name=name,description='Resource-constrained route '+name,
        concept_type='custom',status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.ml.resource_route_execution.resource_route_'+name,
        inputs=[IOSpec(name=inp,type_desc=it)],outputs=[IOSpec(name=out,type_desc=ot)])
        for name,inp,out,it,ot in stages]
    edges=[DependencyEdge(source_id=a[0],target_id=b[0],output_name=a[2],input_name=b[1],
        source_type='object',target_type='object') for a,b in zip(stages,stages[1:])]
    return CDGExport(nodes=nodes,edges=edges,metadata={
        'artifact_source':'competition_generic_execution_reconstruction','publication_status':'draft',
        'source_version_ids':['65947cba-17e3-5a60-ac2b-7a3545885a39'],
        'scope':'Finite directed route planning with nonnegative integer costs and bounded integer resources; exact expanded-state Dijkstra.',
        'choices':['Candidate filtering for closures and resource bounds','Independent path verification before plan selection','Explicit optimal, infeasible, or search_limit result'],
        'exclusions':['Historical winning agent','Arbitrary game or stochastic simulation','Continuous resource dynamics','Partial-route feasibility claim after search limit'],
        'num_nodes':5,'num_edges':4})
