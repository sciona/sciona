"""Draft graph for the complete independent Otto CPU stacking realization."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus


def build_otto_graph():
    nodes=[AlgorithmicNode(node_id=name,name=name,description=description,
        concept_type='custom',status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.ml.otto_execution.otto_'+name,
        inputs=[IOSpec(name=inp,type_desc=it)],outputs=[IOSpec(name=out,type_desc=ot)])
        for name,description,inp,out,it,ot in (
            ('prepare','Validate private populations, fold partitions and explicit control structure','payload','prepared','dict','object'),
            ('execute','Fit all first-level entries, supplemental features, meta selection/refits and final blend','prepared','result','object','dict'))]
    return CDGExport(nodes=nodes,edges=[DependencyEdge(source_id='prepare',target_id='execute',
        output_name='prepared',input_name='prepared',source_type='object',target_type='object')],metadata={
        'artifact_source':'competition_source_corrected_execution','publication_status':'draft',
        'source_version_ids':['dc4969f7-e5a4-5644-a9d0-fa10493458ed'],
        'scope':'Independent Otto CPU realization: 33 first-level entries, seven supplemental blocks, four-fold meta selection, full bags and final blend.',
        'choices':['Shared five-fold first-level predictions and full-reference query refits',
                   'Fixed transductive three-dimensional embedding and explicit five-feature entry',
                   'Raw features added only to neural meta input',
                   '250 XGBoost, 600 Lasagne and 250 AdaBoost-ExtraTrees meta models per bag',
                   'Explicit caller controls, candidate grids and supplemental multiplicities'],
        'exclusions':['Exact historical controls, feature counts, winning accuracy or training volume',
                      'GPU execution, cross-platform or clean-install qualification',
                      'Independent generalization estimate from meta selection loss',
                      'Out-of-sample t-SNE transform for new identities'],
        'num_nodes':2,'num_edges':1})
