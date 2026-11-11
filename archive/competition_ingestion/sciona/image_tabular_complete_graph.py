"""Complete explicit image/tabular lifecycle; feature and model choices stay in the recipe."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,DependencyEdge,IOSpec,NodeStatus


def build_complete_image_tabular_graph():
    stages=[('prepare','payload','dict','object'),('fit','prepared','object','object'),
            ('calibrate','fitted','object','object'),('predict','calibrated','object','dict')]
    bounds=[
        'Finite version-1 JSON with normalized RGB images and aligned numeric/categorical metadata. Disjoint training/calibration/query groups, both binary classes in labeled populations, and group-consistent training folds. Caller owns image meaning, units and group identities.',
        'Immutable prepared configuration. Thirty-one fixed visual descriptors; fold-local median imputation, categorical encoding and fused scaling. CPU float64 dropout MLP with smoothed BCE and full configured epochs; retained fold models, complete OOF coverage and equal probability mean. No full-data refit or learned visual backbone.',
        'Trusted unmodified fitted ensemble. Existing reusable empirical-F1 provider fits a threshold on separate calibration groups, with inclusive comparison and highest-threshold ties. This is decision selection, not probability calibration or independent accuracy.',
        'Trusted calibrated state. Existing reusable threshold provider produces ordered binary decisions from query scores without refitting. Return scores, threshold, counts and aggregate training losses; no named submission format or portable state claim.',
    ]
    nodes=[AlgorithmicNode(node_id=name,name=name,description=bounds[i],concept_type='custom',status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.ml.image_tabular_complete_execution.'+name,
        inputs=[IOSpec(name=inp,type_desc=it,constraints=bounds[i])],
        outputs=[IOSpec(name='result',type_desc=ot,constraints=bounds[i])])
        for i,(name,inp,it,ot) in enumerate(stages)]
    edges=[DependencyEdge(source_id=stages[i-1][0],target_id=name,output_name='result',input_name=inp,
                          source_type='object',target_type='object') for i,(name,inp,_,_) in enumerate(stages) if i]
    return CDGExport(nodes=nodes,edges=edges,metadata=dict(artifact_source='image_tabular_complete_execution',
        catalog_output_sources=dict(result=dict(node_id='predict',output_name='result')),
        reuse_decision='Generic F1 fitting and threshold application remain independently reusable. Image encoding, metadata policy, grouping, folds, descriptors, neural models and equal averaging are explicit recipe choices.',
        limits=['Synthetic computational qualification only; no cross-domain predictive effectiveness or historical accuracy claim.',
                'Features and training populations are materialized in memory; no large-scale resource qualification.',
                'Trusted in-process model states only; source scope and catalog approval remain separate checks.']))
