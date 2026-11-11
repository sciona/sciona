"""Corrected full-grid Connectomics execution candidate."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus


def build_connectomics_graph():
    nodes = [AlgorithmicNode(
        node_id=name, name=name, description='Connectomics ' + name,
        concept_type='custom', status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.ml.connectomics_execution.connectomics_' + name,
        inputs=[IOSpec(name=inp, type_desc=intype)],
        outputs=[IOSpec(name=out, type_desc=outtype)])
        for name, inp, out, intype, outtype in [
            ('prepare', 'payload', 'prepared', 'dict', 'object'),
            ('execute', 'prepared', 'result', 'object', 'dict')]]
    return CDGExport(nodes=nodes, edges=[DependencyEdge(
        source_id='prepare', target_id='execute', output_name='prepared',
        input_name='prepared', source_type='object', target_type='object')], metadata={
        'artifact_source': 'competition_execution_reconstruction',
        'publication_status': 'draft',
        'source_version_id': 'e9f83a4e-4cb1-5297-8c04-8fd65c6e7d33',
        'source_commits': ['b3abb178b9a9d558fdd5a7de49ae50425576a011',
                           '8d04380d474723467b5a717328efd0c9fc5bd898'],
        'scope': 'Complete simple or tuned weighted threshold/filter precision ensemble, with optional strict precedence directivity blend.',
        'runtime_contract': 'Version 1 private nonnegative time-by-node signals; explicit mode and directivity; provisioned hash-verified software via SCIONA_CONNECTOMICS_SOURCE_DIR.',
        'source_corrections': 'Forward scalar thresholds through both sweep and filter, and filter identity to tuned weighting. Preserve all 120 grid entries including duplicates, historical PCA, circular filter boundaries and post-aggregation normalization.',
        'numeric_contract': 'Fortran float32 source input; historical SVD sample-count variance; 80 percent components; optional 0.997 precision plus 0.003 precedence score.',
        'exclusions': ['Original buggy-output parity', 'Historical competition accuracy',
                       'Bayesian averaging guarantee', 'Tier 1 human certification',
                       'Tier 2 usage qualification', 'Degenerate covariance or score ranges'],
        'num_nodes': 2, 'num_edges': 1})
