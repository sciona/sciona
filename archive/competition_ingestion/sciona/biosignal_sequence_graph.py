"""Complete generic five-stage biosignal sequence lifecycle behind validated input and execution boundaries."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,DependencyEdge,IOSpec,NodeStatus


def build_biosignal_sequence_graph():
    nodes=[AlgorithmicNode(node_id=name,name=name,description='Generic Biosignal Sequence '+name,concept_type='custom',status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.ml.biosignal_sequence_execution.biosignal_sequence_'+name,
        inputs=[IOSpec(name=inp,type_desc=it)],outputs=[IOSpec(name=out,type_desc=ot)])
        for name,inp,out,it,ot in [('prepare','payload','prepared','dict','object'),('execute','prepared','result','object','dict')]]
    return CDGExport(nodes=nodes,edges=[DependencyEdge(source_id='prepare',target_id='execute',output_name='prepared',input_name='prepared',source_type='object',target_type='object')],metadata={
        'artifact_source':'competition_generic_execution_reconstruction','publication_status':'draft',
        'source_version_ids':['1587d081-d47c-5595-8bed-d3d3aa6465af'],
        'scope':'Generic binary signal classification: subject-separated windows, log PSD, waveform and spectrogram CNNs, spectral masking and balanced loss, recording aggregation and F1 threshold.',
        'choices':['Explicit shared sample rate and aligned channel order','Periodic Hann one-sided PSD with segment-mean detrending','Training-only recording-balanced channel scaling','CPU float64 Adam with class/recording-balanced loss and spectral masking','Mean window probability per recording and separate highest-threshold F1 calibration'],
        'exclusions':['Clinical accuracy or historical winning recipe','Automatic subject identity or montage discovery','Incomplete trailing window/FFT samples','Large-scale resource and cross-platform qualification'],
        'num_nodes':2,'num_edges':1})
