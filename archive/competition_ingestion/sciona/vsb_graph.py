"""Draft graph for the independent full VSB signal-to-decision pipeline."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,DependencyEdge,IOSpec,NodeStatus


def build_vsb_graph():
    nodes=[AlgorithmicNode(node_id=name,name=name,description=description,concept_type='custom',status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.ml.vsb_execution.vsb_'+name,
        inputs=[IOSpec(name=inp,type_desc=it)],outputs=[IOSpec(name=out,type_desc=ot)])
        for name,description,inp,out,it,ot in (
            ('prepare','Validate private raw signal triples, labels, identities and complete fold lifecycle','payload','prepared','dict','object'),
            ('execute','Extract phase-resolved peak features, fit 125 trees and threshold signal decisions','prepared','result','object','dict'))]
    return CDGExport(nodes=nodes,edges=[DependencyEdge(source_id='prepare',target_id='execute',output_name='prepared',input_name='prepared',source_type='object',target_type='object')],metadata={
        'artifact_source':'competition_source_corrected_execution','publication_status':'draft',
        'source_version_ids':['d79e40e4-45b0-5e74-a83c-c68c91f12afb'],
        'scope':'Independent complete VSB recursive baseline, custom peaks, phase aggregates and repeated LightGBM signal decisions.',
        'choices':['Floating-point baseline filter, directional peak intersection and source negative knee slicing',
            'Actual-length clipped descriptors, FFT fundamental as one cycle per supplied signal, right-closed phase quadrants',
            'Complete nine-feature aggregation retains NaN missing groups and full source filtering',
            '25 repetitions of five random folds; both heldouts influence early stopping',
            'Signal-level threshold from in-training predictions; strict greater-than query decisions broadcast over triples',
            'Current single-thread deterministic CPU LightGBM; explicit rejection of undefined peak or fold inputs'],
        'exclusions':['Unsupported intake wavelet, matrix transform, deep network and weighted tree/deep blend',
            'Historical native precision, untouched-test validation, competitive accuracy or production qualification'],
        'num_nodes':2,'num_edges':1})
