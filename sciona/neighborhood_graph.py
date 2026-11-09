"""Native single-atom graph for explicit numeric neighborhood blending."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, IOSpec, NodeStatus
from sciona.atoms.ml.neighborhoods import blend_neighbor_means


def build_neighborhood_graph():
    return CDGExport(nodes=[AlgorithmicNode(node_id='blend',name='Blend neighbor means',
        description='Blend signed values using explicit caller-selected neighborhoods.',
        concept_type='custom',status=NodeStatus.ATOMIC,
        matched_primitive=blend_neighbor_means.__module__+'.'+blend_neighbor_means.__name__,
        inputs=[IOSpec(name='values',type_desc='numpy.ndarray',constraints='Nonempty finite real vector; signed values allowed.'),
                IOSpec(name='neighbors',type_desc='numpy.ndarray',constraints='Square boolean mask aligned to values; rows select neighbors; self-selection allowed.'),
                IOSpec(name='strength',type_desc='float',constraints='Finite Python scalar in [0,1]; bool excluded.')],
        outputs=[IOSpec(name='result',type_desc='numpy.ndarray',constraints='Fresh finite float64 vector. Empty neighbor rows retain original values. No clipping or general sum conservation.')])],
        edges=[],metadata=dict(publication_status='draft',scope='Domain-independent neighborhood blending with explicit adjacency.'))
