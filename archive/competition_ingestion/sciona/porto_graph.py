"""Draft graph for the independent Porto five-neural/one-tree ensemble."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,DependencyEdge,IOSpec,NodeStatus


def build_porto_graph():
    nodes=[AlgorithmicNode(node_id=name,name=name,description=description,concept_type='custom',status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.ml.porto_execution.porto_'+name,
        inputs=[IOSpec(name=inp,type_desc=it)],outputs=[IOSpec(name=out,type_desc=ot)])
        for name,description,inp,out,it,ot in (
            ('prepare','Validate private disjoint populations and explicit ensemble configuration','payload','prepared','dict','object'),
            ('execute','Fit five transductive DAE neural branches and raw prepared tree; average probabilities','prepared','result','object','dict'))]
    return CDGExport(nodes=nodes,edges=[DependencyEdge(source_id='prepare',target_id='execute',output_name='prepared',input_name='prepared',source_type='object',target_type='object')],metadata={
        'artifact_source':'competition_source_corrected_execution','publication_status':'draft',
        'source_version_ids':['ad3239bf-5a38-565a-81da-3d3cba9a02b0'],
        'scope':'Independent transductive Porto ensemble with five DAE neural models and one LightGBM model.',
        'choices':['Explicit column roles, joint one-hot population and midpoint average-tie RankGauss',
            'Same-column swap-noise, clean reconstruction targets, caller-selected hidden features',
            'Explicit SGD momentum and dropout scaling; neural models receive latent features only',
            'Tree receives prepared raw features; final six probabilities have equal weights'],
        'exclusions':['Exact historical five-model configurations, optimizer, CV and checkpoint replay',
            'Historical competitive accuracy, GPU or production qualification',
            'Unsubstantiated intake imputation, raw-plus-latent concatenation, XGBoost and rank blending'],
        'num_nodes':2,'num_edges':1})
