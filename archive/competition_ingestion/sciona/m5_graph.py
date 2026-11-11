"""Draft source-corrected graph for independent M5 preprocessing and forecasting."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,DependencyEdge,IOSpec,NodeStatus


def build_m5_graph():
    nodes=[AlgorithmicNode(node_id=name,name=name,description=description,concept_type='custom',status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.ml.m5_execution.m5_'+name,
        inputs=[IOSpec(name=inp,type_desc=it)],outputs=[IOSpec(name=out,type_desc=ot)])
        for name,description,inp,out,it,ot in (
            ('prepare','Validate private history and explicit hierarchy, price and calendar roles','payload','prepared','dict','object'),
            ('execute','Construct features, train six pooled families and forecast 28 days with recursive feedback','prepared','result','object','dict'))]
    return CDGExport(nodes=nodes,edges=[DependencyEdge(source_id='prepare',target_id='execute',output_name='prepared',input_name='prepared',source_type='object',target_type='object')],metadata={
        'artifact_source':'competition_source_corrected_execution','publication_status':'draft',
        'source_version_ids':['f64bac1a-ca7f-5f29-a645-ec7b2b42ca4b'],
        'scope':'Independent M5 raw semantic-role preprocessing, six pooled Tweedie families, full recursive/nonrecursive horizon and arithmetic mean.',
        'choices':['Caller-supplied hierarchy, category, price and calendar roles; source precision selection and release filtering',
            'Fixed cutoff target encodings and family-specific feature order; full 3000-round learner controls',
            'Three pooling granularities for each forecast mode; 220 models when supplied the qualified synthetic pool inventory',
            'Independent deterministic CPU training; recursive diagnostics overlap training and do not imply unbiased validation',
            'Separate family feedback using last100days and daily temporary means; 28-day forecasts and equal six-way mean'],
        'exclusions':['Unestablished final reconciliation or multiplicative adjustment intake claims',
            'Historical sample-volume, hardware/RNG/weight parity, competitive accuracy or production qualification'],
        'num_nodes':2,'num_edges':1})
