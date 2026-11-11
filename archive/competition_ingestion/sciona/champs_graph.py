"""Complete corrected CHAMPS competition execution graph, draft pending review."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,DependencyEdge,IOSpec,NodeStatus


def build_champs_graph():
    nodes=[AlgorithmicNode(node_id=name,name=name,description='CHAMPS '+name,
        concept_type='custom',status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.ml.champs_execution.champs_'+name,
        inputs=[IOSpec(name=inp,type_desc=intype)],outputs=[IOSpec(name=out,type_desc=outtype)])
        for name,inp,out,intype,outtype in [('prepare','payload','prepared','dict','object'),
                                          ('execute','prepared','result','object','dict')]]
    return CDGExport(nodes=nodes,edges=[DependencyEdge(source_id='prepare',target_id='execute',
        output_name='prepared',input_name='prepared',source_type='object',target_type='object')],
        metadata={'artifact_source':'competition_execution_reconstruction','publication_status':'draft',
            'source_version_id':'af2888ee-499c-5db7-8576-f4096e1f8804',
            'source_commits':['4a42e18b5b88043fb40ec15289216a1d88789698'],
            'scope':'Raw XYZ chemistry, atom/bond/triplet/quadruplet features, fitted categorical vocabulary and target scaling, all thirteen full configured graph transformers, source training/selection, complete state reload, unscaled prediction and coupling-specific central-five ensemble.',
            'runtime_contract':'Version 1 private raw populations with explicit per-model training schedules and full/validation selection. Provisioned source via SCIONA_CHAMPS_SOURCE_DIR; CPU execution and temporary private checkpoints.',
            'source_corrections':'Correct triplet packer/model column offsets; modern OpenBabel/pandas interfaces; skip backward during chunked evaluation; reserve eight supervised vocabulary IDs. Source graph transformers replace inaccurate SchNet/LightGBM intake claims.',
            'numeric_contract':'Per-subtype mean/sample-standard-deviation unscaling, six-decimal per-model source boundary, coupling-specific nine-model selection and central-five average.',
            'exclusions':['Original pretrained checkpoint prediction parity','Historical competition accuracy','Record-specific original chemistry correction table','Tier 1 human certification','Tier 2 usage qualification'],
            'num_nodes':2,'num_edges':1})
