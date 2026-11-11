"""Draft source-corrected graph for the complete independent Amex ensemble."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,DependencyEdge,IOSpec,NodeStatus


def build_amex_graph():
    nodes=[AlgorithmicNode(node_id=name,name=name,description=description,concept_type='custom',status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.ml.amex_execution.amex_'+name,
        inputs=[IOSpec(name=inp,type_desc=it)],outputs=[IOSpec(name=out,type_desc=ot)])
        for name,description,inp,out,it,ot in (
            ('prepare','Validate private customer sequences, raw category roles and complete fold populations','payload','prepared','dict','object'),
            ('execute','Preprocess raw sequences, fit row/manual/neural ensembles and apply literal four-way blend','prepared','result','object','dict'))]
    return CDGExport(nodes=nodes,edges=[DependencyEdge(source_id='prepare',target_id='execute',output_name='prepared',input_name='prepared',source_type='object',target_type='object')],metadata={
        'artifact_source':'competition_source_corrected_execution','publication_status':'draft',
        'source_version_ids':['d505faaa-7277-5559-b32c-f95c9f1f9d63'],
        'scope':'Independent full Amex raw preprocessing, manual/sequence features, grouped row DART, downstream trees and neural branches.',
        'choices':['Explicit runtime category/month/time/fill roles; numeric flooring and source aggregate quantization',
            'Joint vocabulary and month ranks; timestamp-rank windows; source greedy bin omission and missing codes',
            'Customer-stratified row cross-fit predictions in left-padded slots; right-padded neural sequence masks',
            'Fifteen DART models at 4500 configured rounds and ten neural models at ten epochs',
            'Current deterministic CPU learners, effective Adam schedule, strict Amex-metric checkpoint selection',
            'Literal final weights .30/.35/.15/.10 sum to .90 without normalization'],
        'exclusions':['Unsupported XGBoost/CatBoost or rank-average intake claims',
            'Historical GPU/RNG/native parity, unbiased nested validation, competitive accuracy or production qualification'],
        'num_nodes':2,'num_edges':1})
