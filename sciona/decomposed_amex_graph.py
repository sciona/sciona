"""Complete independent Amex training with explicit stages and reusable combination."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus


def build_decomposed_amex_graph():
    recipe='sciona.atoms.ml.decomposed_amex_execution.'
    definitions=[
        ('prepare','Validate raw sequences and fold populations','sciona.atoms.ml.amex_execution.amex_prepare',[('payload','dict')],[('prepared','object')]),
        ('populations','Prepare ordered joint populations',recipe+'populations',[('prepared','object')],[('population','dict')]),
        ('features','Build complete manual, sequence and row features',recipe+'features',[('population','dict')],[('feature_values','dict')]),
        ('row_models','Fit five grouped row models',recipe+'row_models',[('population','dict'),('feature_values','dict')],[('row_result','dict')]),
        ('assembly','Assemble manual, prediction and normalized features',recipe+'assembly',[('feature_values','dict'),('row_result','dict')],[('assembled','dict')]),
        ('tree_models','Fit ten downstream tree models',recipe+'tree_models',[('population','dict'),('assembled','dict')],[('tree_result','dict')]),
        ('neural_models','Fit ten neural models',recipe+'neural_models',[('population','dict'),('feature_values','dict'),('assembled','dict')],[('neural_result','dict')]),
        ('combination_inputs','Order source branches and literal weights',recipe+'combination_inputs',[('tree_result','dict'),('neural_result','dict')],[('values','numpy.ndarray'),('coefficients','numpy.ndarray')]),
        ('combine','Compute reusable ordered weighted sum','sciona.atoms.numerical.statistics.ordered_sum.ordered_weighted_sum',[('values','numpy.ndarray'),('coefficients','numpy.ndarray')],[('scores','numpy.ndarray')]),
        ('result','Encode complete recipe result',recipe+'result',[('scores','numpy.ndarray')],[('result','dict')]),
    ]
    nodes=[AlgorithmicNode(node_id=key,name=description,description=description,concept_type='custom',
        status=NodeStatus.ATOMIC,matched_primitive=fqdn,inputs=[IOSpec(name=n,type_desc=t) for n,t in inputs],
        outputs=[IOSpec(name=n,type_desc=t) for n,t in outputs]) for key,description,fqdn,inputs,outputs in definitions]
    producers={p.name:(n.node_id,p.type_desc) for n in nodes for p in n.outputs}
    edges=[]
    for node in nodes:
        for port in node.inputs:
            if port.name=='payload':continue
            source,kind=producers[port.name]
            edges.append(DependencyEdge(source_id=source,target_id=node.node_id,output_name=port.name,
                input_name=port.name,source_type=kind,target_type=port.type_desc))
    return CDGExport(nodes=nodes,edges=edges,metadata=dict(
        artifact_source='competition_source_corrected_execution',publication_status='draft',
        source_version_ids=['d505faaa-7277-5559-b32c-f95c9f1f9d63'],
        scope='Complete independent Amex raw preprocessing, nine manual blocks, grouped row DART, downstream tree/neural variants and literal nonnormalized blend.',
        reuse_boundary='Ordered weighted sum is domain independent. Feature choices, learner controls, branch order and coefficients remain recipe policy.',
        catalog_output_sources={'result':dict(node_id='result',output_name='result')},
        num_nodes=len(nodes),num_edges=len(edges)))
