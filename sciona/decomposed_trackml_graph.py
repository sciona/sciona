"""Draft geometry-injected tracking graph; further decomposition remains required."""
import inspect
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,DependencyEdge,EdgeKind,IOSpec,NodeStatus
from sciona.atoms.particle_tracking.helix_geometry import atoms as geometry
from sciona.atoms.physics import trackml_execution as boundary
from sciona.atoms.physics.decomposed_trackml_execution import track


def build_decomposed_trackml_graph():
    nodes=[];edges=[]
    def add(name,fn,outputs):
        signature=inspect.signature(fn)
        numeric=fn.__module__==geometry.__name__
        inputs=[IOSpec(name=p.name,type_desc=('float' if p.default is not p.empty else 'ndarray') if numeric else ('dict' if p.annotation is dict else 'object'),
            required=p.default is p.empty,default_value_repr='' if p.default is p.empty else repr(p.default)) for p in signature.parameters.values()]
        nodes.append(AlgorithmicNode(node_id=name,name=name,description=inspect.getdoc(fn) or name,
            concept_type='custom',status=NodeStatus.ATOMIC,matched_primitive=fn.__module__+'.'+fn.__name__,inputs=inputs,
            outputs=[IOSpec(name=n,type_desc='ndarray' if numeric else ('dict' if name=='encode' else 'object')) for n in outputs]))
    add('circle',geometry.circle_from_three_points,['xm','ym','radius'])
    add('pitch_two',geometry.helix_pitch_from_two_points,['pitch','phase','dz'])
    add('pitch_three',geometry.helix_pitch_least_squares,['pitch','phase','dz','loss'])
    add('decode',boundary.trackml_decode,['prepared'])
    add('track',track,['assignments'])
    add('encode',boundary.trackml_encode,['result'])
    for source,output,target,input_name,kind in [
        ('circle','xm','track','circle_function',EdgeKind.CALLABLE_INJECTION),
        ('pitch_two','pitch','track','pitch_two_function',EdgeKind.CALLABLE_INJECTION),
        ('pitch_three','pitch','track','pitch_three_function',EdgeKind.CALLABLE_INJECTION),
        ('decode','prepared','track','prepared',EdgeKind.DATA_FLOW),
        ('track','assignments','encode','assignments',EdgeKind.DATA_FLOW)]:
        edges.append(DependencyEdge(source_id=source,target_id=target,output_name=output,input_name=input_name,
            source_type='object',target_type='object',edge_kind=kind))
    return CDGExport(nodes=nodes,edges=edges,metadata=dict(publication_status='draft',
        artifact_source='tracking_geometry_decomposition',source_version_id='11c41ece-c648-50b4-b211-7e319ba36dea',
        catalog_output_sources={'result':dict(node_id='encode',output_name='result')},
        num_nodes=6,num_edges=5,
        scope='Every configured tracking round with three separately bound reusable geometry providers.',
        limits=['Further detector, neighbor, intersection and recipe-stage decomposition remains pending.',
                'Existing geometry providers require exact canonical interface qualification before graph publication.',
                'Caller-supplied maps are applied; learned calibration and historical accuracy are not established.']))
