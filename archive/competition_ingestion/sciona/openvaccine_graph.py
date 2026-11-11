"""Draft graph for the corrected DasLab OpenVaccine reconstruction."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,DependencyEdge,IOSpec,NodeStatus


def build_openvaccine_graph():
    nodes=[AlgorithmicNode(node_id=name,name=name,description='OpenVaccine '+name,concept_type='custom',status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.ml.openvaccine_execution.openvaccine_'+name,
        inputs=[IOSpec(name=inp,type_desc=intype)],outputs=[IOSpec(name=out,type_desc=outtype)])
        for name,inp,out,intype,outtype in [('prepare','payload','prepared','dict','object'),('execute','prepared','result','object','dict')]]
    return CDGExport(nodes=nodes,edges=[DependencyEdge(source_id='prepare',target_id='execute',output_name='prepared',input_name='prepared',source_type='object',target_type='object')],metadata={
        'artifact_source':'competition_execution_reconstruction','publication_status':'draft',
        'source_version_ids':['9510a335-f191-5642-9ae1-20459e7d4eba','bc2e55f3-60b8-56a3-8519-87f31ed27246'],
        'source_commits':['b95af835cc38bd7dc55ed6e7397d124787f86996','64f01d52f8fcba2b88c0011d56360554b9a95012'],
        'scope':'Pinned folding and features, shared autoencoder pretraining, twenty explicitly split models, teacher-driven supervised/PL refinement, validation rollback and checkpoint-backed reverse-averaged ensemble.',
        'source_corrections':'All-member clipping; per-record feature batching; strict folded-input boundaries. Explicit reconstructed training recipe and caller-defined populations.',
        'exclusions':['Original winning clustered-final-blend equivalence','Historical accuracy','Independent CV accuracy','Stochastic checkpoint replay','Tier1 human certification','Tier2 usage qualification'],
        'num_nodes':2,'num_edges':1})
