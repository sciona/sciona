"""Complete analytical tracking realization of the competition source flow."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, IOSpec, NodeStatus

SOURCE_VERSION = '11c41ece-c648-50b4-b211-7e319ba36dea'
SOURCE_HASH = 'c0b732d94cd161a98dfc8a24a70a1805453a3de2d5b4d71ee7719fda4c639183'
PRIMITIVE = 'sciona.atoms.particle_tracking.source_tracking.atoms.find_tracks'


def build_tracking_source_execution():
    inputs = [
        IOSpec(name='detector_modules', type_desc='numpy.ndarray', constraints='Nonempty finite real (N,6): volume_id, layer_id, cx, cy, cz, module_hv. Source-compatible cylinders/caps; positive half lengths.'),
        IOSpec(name='observations', type_desc='numpy.ndarray', constraints='Nonempty finite real (M,6): volume_id, layer_id, module_id, x, y, z. Integral nonnegative identifiers <=32767; coordinates finite after float32 rounding.'),
    ]
    for name, default, bound in [('extension_steps', 11, '2..100'), ('commitment_rounds', 1, '1..100'), ('commitment_limit', 1000, '1..1000000')]:
        inputs.append(IOSpec(name=name, type_desc='int', constraints='Integer '+bound+'; excludes bool', required=False, default_value_repr=str(default)))
    return CDGExport(nodes=[AlgorithmicNode(node_id='tracking', name='Reconstruct tracks with original analytical flow',
        description='Pinned original detector construction, seeding, extension, fitting, pairing, ranking, pruning and commitment; no cell/calibrated paths.',
        concept_type='custom', status=NodeStatus.ATOMIC, matched_primitive=PRIMITIVE, inputs=inputs,
        outputs=[IOSpec(name='track_labels', type_desc='numpy.ndarray', constraints='int64 shape (M,), aligned with observations; zero means unassigned')])],
        edges=[], metadata={'artifact_source': 'competition_source_execution', 'publication_status': 'draft',
                           'source_version_id': SOURCE_VERSION, 'source_content_hash': SOURCE_HASH, 'num_nodes': 1, 'num_edges': 0,
                           'limitations': 'Analytical observation-only flow. No accuracy, calibration, cell-feature or scoring claim.'})
