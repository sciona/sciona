"""Five explicit lifecycle stages for the selected geotemporal regression recipe."""
import inspect
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus
import sciona.atoms.ml.geotemporal_complete_execution as adapters

STAGES=('prepare','features','fit','predict','finish')


def build_complete_geotemporal_graph():
    nodes=[]
    for name in STAGES:
        fn=getattr(adapters,name);signature=inspect.signature(fn)
        nodes.append(AlgorithmicNode(node_id=name,name=name,description=inspect.getdoc(fn),
            concept_type='custom',status=NodeStatus.ATOMIC,matched_primitive=adapters.__name__+'.'+name,
            inputs=[IOSpec(name=p.name,type_desc=inspect.formatannotation(p.annotation)) for p in signature.parameters.values()],
            outputs=[IOSpec(name='result',type_desc=inspect.formatannotation(signature.return_annotation))]))
    edges=[DependencyEdge(source_id=a.node_id,target_id=b.node_id,output_name='result',input_name=b.inputs[0].name,
        source_type='object',target_type='object') for a,b in zip(nodes,nodes[1:])]
    return CDGExport(nodes=nodes,edges=edges,metadata=dict(publication_status='draft',
        scope='Explicit causal planar geotemporal regression with forward validation, full refit and same-time neighbor postprocessing.',
        reuse_decision='Neighborhood mean blending is independently reusable; metric coordinates, clock, lag rules, validation populations and clipping remain recipe policy.',
        limits=['Trusted in-process states only; no portable fitted-state persistence.',
            'Forward-time validation does not establish unseen-location accuracy or spatial holdout independence.',
            'Query smoothing depends on the same-time query population; no named competition format or historical accuracy claim.']))
