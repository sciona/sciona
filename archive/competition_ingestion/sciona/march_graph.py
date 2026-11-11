"""Generic season aggregation, graph/rating features and calibrated classifier."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,DependencyEdge,IOSpec,NodeStatus


def build_march_graph():
    nodes=[AlgorithmicNode(node_id=name,name=name,description='Generic March '+name,concept_type='custom',status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.ml.march_execution.march_'+name,
        inputs=[IOSpec(name=inp,type_desc=it)],outputs=[IOSpec(name=out,type_desc=ot)])
        for name,inp,out,it,ot in [('prepare','payload','prepared','dict','object'),('execute','prepared','result','object','dict')]]
    return CDGExport(nodes=nodes,edges=[DependencyEdge(source_id='prepare',target_id='execute',output_name='prepared',input_name='prepared',source_type='object',target_type='object')],metadata={
        'artifact_source':'competition_generic_execution_reconstruction','publication_status':'draft',
        'source_version_ids':['26d177bb-7d6d-5ee1-b881-2bafe69064cd'],
        'scope':'Aggregate box-score efficiencies, weighted loser-to-winner PageRank, chronological Elo, paired features and logistic fitting with held-out sigmoid calibration.',
        'choices':['Explicit possessions and rating controls','Strictly ordered disjoint fitting, calibration and prediction seasons','Mirrored fitting/calibration rows and orientation-symmetric prediction'],
        'exclusions':['Historical winning recipe or performance','True strength-of-schedule certification','Empirical calibration or generalization guarantee','XGBoost alternative','Tier1 or Tier2 certification'],
        'num_nodes':2,'num_edges':1})
