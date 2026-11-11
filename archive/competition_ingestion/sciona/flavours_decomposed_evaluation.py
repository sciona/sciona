"""Complete evaluated Flavours graph with three explicit reused metric nodes."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,DependencyEdge,IOSpec,NodeStatus
from sciona.flavours_decomposed_pipeline import build_pipeline_graph


def build_evaluated_graph():
    pipeline=build_pipeline_graph();nodes=list(pipeline.nodes);edges=list(pipeline.edges)
    array='numpy.ndarray';base='sciona.atoms.ml.'
    def node(name,primitive,inputs,outputs,description):
        nodes.append(AlgorithmicNode(node_id=name,name=name,concept_type='custom',status=NodeStatus.ATOMIC,
            matched_primitive=base+primitive,description=description,
            inputs=[IOSpec(name=n,type_desc=t) for n,t in inputs],outputs=[IOSpec(name=n,type_desc=t) for n,t in outputs]))
    def edge(source,output,target,port,kind=array):
        edges.append(DependencyEdge(source_id=source,output_name=output,target_id=target,input_name=port,source_type=kind,target_type=kind))
    node('evaluation_inputs','flavours_evaluation_wiring.evaluation_inputs',
        [('query_inputs','list'),('controls','dict'),('labels',array)],
        [('merged_query','list'),('sizes','list'),('weights_a',array),('weights_b',array),('correlation_mass',array),('quality',array),('neighbours','int'),('step','int')],
        'Validate controls before fitting; merge physical populations only, keeping evaluation metadata separate.')
    edge('evaluation_inputs','merged_query','inputs','query_inputs','list')
    node('population_scores','flavours_evaluation_wiring.split_population_scores',[('scores',array),('sizes','list')],
        [('query',array),('agreement_a',array),('agreement_b',array),('correlation',array)],'Restore caller population boundaries without sorting or changing scores.')
    edge('query_blend','scores','population_scores','scores');edge('evaluation_inputs','sizes','population_scores','sizes','list')
    node('discrimination_inputs','flavours_evaluation_wiring.discrimination_inputs',
        [('labels',array),('scores',array),('quality',array)],[('labels',array),('scores',array)],
        'Apply strict quality > .4 selection for diagnostic OOF AUC, never for fitting.')
    edge('inputs','labels','discrimination_inputs','labels');edge('oof_blend','scores','discrimination_inputs','scores')
    edge('evaluation_inputs','quality','discrimination_inputs','quality')
    node('agreement_metric','constrained_ml.decorrelation.atoms.compute_ks_agreement',
        [('data_predictions',array),('mc_predictions',array),('weights_data',array),('weights_mc',array)],[('ks','float')],
        'Reuse the registered weighted KS metric with evaluation-only weights.')
    for output,port in [('agreement_a','data_predictions'),('agreement_b','mc_predictions')]:edge('population_scores',output,'agreement_metric',port)
    for output,port in [('weights_a','weights_data'),('weights_b','weights_mc')]:edge('evaluation_inputs',output,'agreement_metric',port)
    node('correlation_metric','constrained_ml.decorrelation.atoms.compute_cvm_mass_decorrelation',
        [('predictions',array),('protected_variable',array),('n_neighbours','int'),('step','int')],[('cvm','float')],
        'Reuse CvM with 200 neighbours and step 50; protected mass never enters training.')
    edge('population_scores','correlation','correlation_metric','predictions')
    edge('evaluation_inputs','correlation_mass','correlation_metric','protected_variable')
    edge('evaluation_inputs','neighbours','correlation_metric','n_neighbours','int');edge('evaluation_inputs','step','correlation_metric','step','int')
    node('discrimination_metric','constrained_ml.decorrelation.atoms.roc_auc_truncated_weighted',
        [('labels',array),('predictions',array)],[('auc','float')],'Reuse source-weighted truncated AUC with its fixed default thresholds and weights.')
    edge('discrimination_inputs','labels','discrimination_metric','labels');edge('discrimination_inputs','scores','discrimination_metric','predictions')
    node('report','flavours_evaluation_wiring.diagnostic_report',
        [('query_scores',array),('ks','float'),('cvm','float'),('auc','float'),('sizes','list')],[('result','dict')],
        'Retain metric failures as failures; no model selection or score modification.')
    edge('population_scores','query','report','query_scores');edge('evaluation_inputs','sizes','report','sizes','list')
    for source,port in [('agreement_metric','ks'),('correlation_metric','cvm'),('discrimination_metric','auc')]:edge(source,port,'report',port,'float')
    return CDGExport(nodes=nodes,edges=edges,metadata=dict(publication_status='draft',target_tier=3,
        num_nodes=len(nodes),num_edges=len(edges),scope='Complete 115-fit physical-input workflow, query/control prediction and three explicit source diagnostics.',
        limitations=['Caller establishes consistent physical units, source column meanings and population disjointness.',
            'Quality selection is materialized at runtime; its selected row count is data dependent.',
            'Upstream mass features are not nested in downstream folds; OOF discrimination is diagnostic.',
            'Exactly 200 correlation rows yield a vacuous whole-population CvM window.',
            'No historical engine parity or predictive-quality claim. Fit histories remain in execution traces.']))
