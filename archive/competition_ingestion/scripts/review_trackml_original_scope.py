"""Record reviewed original-stage semantics and explicit cross-domain boundaries."""
import ast
import json
import subprocess

from scripts import audit_decomposed_trackml_readiness as baseline
from scripts.audit_trackml_cell_catalog_readiness import STAGES


# These are reviewed semantic statements, not conclusions inferred from symbol existence.
# Anchors and immutable comparison evidence make the review reproducible and invalidate drift.
STAGE_REVIEW = {
    'detector_geometry_autodiscovery': (
        [('trackml_detector', 'CapsSpec.__init__'), ('trackml_detector', 'CylindersSpec.__init__')],
        ['detector'],
        'Discover layer dimensions within the source detector topology using radial Lloyd clustering and axial mean shift; retain ordering, coverage and gap rules.',
        'Fixed volume groups, seven radial rings and exactly three gaps; not arbitrary detector discovery. Clustering remains independently reusable.'),
    'coordinate_rescaling_for_knn': (
        [('trackml_neighbors', 'CylinderNeighbors.Cylinder.transformToNeighborCoords'), ('trackml_neighbors', 'CapNeighbors.Cap.transformToNeighborCoords')],
        ['neighbors'],
        'Project onto mean-radius cylinders with configured transverse scaling, or onto the layer plane, before Euclidean neighbor search.',
        'Projection singularities are outside the finite-input contract. Layer grouping and source scale selection remain recipe policy.'),
    'per_layer_knn_search': (
        [('trackml_neighbors', 'Neighbors'), ('trackml_neighbors', 'CylinderNeighbors'), ('trackml_neighbors', 'CapNeighbors')],
        ['neighbors', 'runtime'],
        'Fit each layer index and perform configured k-neighbor and radius queries, preserving grouping and source result ordering.',
        'Sparse or exhausted layers reject explicitly; fitted index objects are in-process objects, not portable serialized models.'),
    'circle_from_three_points': (
        [('trackml_geometry', '_source_circle'), ('decomposed_trackml_geometry', 'circle')],
        ['circle'],
        'Fit the circumcircle, retaining source large-radius tangent handling for supported collinear or coincident configurations.',
        'Undefined zero tangent geometry rejects instead of certifying source nonfinite output. The large-radius fallback is an approximation.'),
    'helix_pitch_least_squares': (
        [('trackml_helix', '_pitch_two'), ('trackml_helix', '_pitch_three'), ('decomposed_trackml_geometry', 'pitch_three')],
        ['helix'],
        'Use source phase unwrapping and two-point or three-point axial regression with residual quality and explicit zero-pitch replacement.',
        'Zero phase variance rejects; the tracking recipe supplies its 0.001 replacement policy rather than making it a universal physical constant.'),
    'helix_cylinder_intersection': (
        [('trackml_intersections', 'CylinderIntersector.intersectHelices')],
        ['intersections', 'layer_integration'],
        'Compute candidate cylinder intersections, select forward motion, apply axial bounds and optional correction, and retry later layers under source recursion rules.',
        'Preserve bounded retry, recursive default pre-move behavior and explicit miss sentinels; no guarantee that a future layer is reached.'),
    'helix_cap_intersection': (
        [('trackml_intersections', 'CapIntersector.intersectHelices')],
        ['intersections', 'layer_integration'],
        'Intersect axial planes and apply radial coverage, ring-gap, overlap and optional correction policy with bounded later-layer retries.',
        'Detector coverage and retry rules belong to the recipe; generic plane intersection remains separately selectable.'),
    'perturbative_helix_correction': (
        [('trackml_corrections', 'HelixCorrector.correctCapIntersections'), ('trackml_corrections', 'HelixCorrector.correctCylinderIntersections')],
        ['corrections', 'layer_integration'],
        'Apply supplied per-layer interpolated correction maps with source orientation and curvature scaling, masks, missing-value handling and geometry guards.',
        'The immutable stage describes applying learned maps. Map training is not part of this execution; supplied maps and calibration validity remain caller responsibilities.'),
    'helix_nearest_point_distance': (
        [('trackml_nearest', '_nearest'), ('decomposed_trackml_nearest', 'nearest')],
        ['nearest'],
        'Compute the source fixed-iteration Newton approximation to nearest helix position and distance, with three iterations in the tracking recipe.',
        'No global minimum or convergence guarantee; finite qualified outputs only. Optional phase/arc outputs are outside this graph boundary.'),
    'bayesian_neighbor_evaluation': (
        [('trackml_bayesian', 'BayesianNeighborScorer')],
        ['bayesian', 'extension'],
        'Combine tangent residuals, supplied uncertainty information and local background-density policy to classify neighboring candidates.',
        'Source thresholds and uncertainty coefficients remain explicit recipe policy; missing-map behavior is preserved, not statistical calibration certification.'),
    'greedy_track_commit': (
        [('trackml_candidates', 'zeroUsedHits'), ('trackml_candidates', 'Candidates.submit'), ('trackml_lifecycle', 'TrackLifecycle.findTracks')],
        ['candidates', 'lifecycle'],
        'Rank candidates by descending recipe quality, then greedily claim available observations under configured minimum-count and loss bounds across commit rounds.',
        'The reusable commit primitive consumes caller ordering and supports reserve-skipped policy; the source lifecycle supplies reserve_skipped=False. No global assignment optimum is claimed.'),
}


def anchor(module, qualified_name):
    path = baseline.ROOT/'sciona'/f'{module}.py'
    node = ast.parse(path.read_text())
    for name in qualified_name.split('.'):
        matches = [item for item in node.body if isinstance(item, (ast.ClassDef, ast.FunctionDef)) and item.name == name]
        if len(matches) != 1:
            raise ValueError('Source anchor missing or ambiguous: '+qualified_name)
        node = matches[0]
    return dict(file=str(path.relative_to(baseline.ROOT)), symbol=qualified_name,
        line=node.lineno, file_sha256=baseline.sha(path))


def review():
    state = baseline.audit()
    if state['implementation_drift']:
        raise ValueError('Retained source drift')
    if set(STAGE_REVIEW) != set(STAGES) or set(STAGES) != {s['stage_id'] for s in state['source_stages']}:
        raise ValueError('All immutable original stages must be reviewed')
    evidence_path = 'docs/reviews/trackml_original_component_revalidation.json'
    raw = (baseline.ROOT/evidence_path).read_bytes()
    if raw != subprocess.check_output(['git', 'show', 'c4b6c5ae:'+evidence_path], cwd=baseline.ROOT):
        raise ValueError('Immutable reexecution evidence changed')
    evidence = json.loads(raw)
    if not evidence['passed'] or evidence['ready_nodes'] != 28 or not evidence['exact_published_adapter_dependencies']:
        raise ValueError('Complete reexecution and catalog evidence required')
    graph_path = baseline.ROOT/'docs/reviews/decomposed_trackml_cell_graph_execution.json'
    if baseline.sha(graph_path) != evidence['graph_evidence_sha256']:
        raise ValueError('Reexecuted graph evidence changed')
    for relative, digest in json.loads(graph_path.read_text())['implementation_sha256'].items():
        if baseline.sha(baseline.ROOT/relative) != digest:
            raise ValueError('Reexecuted graph implementation changed')
    for name, component in evidence['components'].items():
        if baseline.sha(baseline.ROOT/f'docs/reviews/competition_trackml_{name}.json') != component['report_sha256']:
            raise ValueError('Retained comparison report changed')
        if baseline.sha(baseline.ROOT/f'scripts/validate_trackml_{name}.py') != component['validator_sha256']:
            raise ValueError('Retained comparison validator changed')
    reviewed = []
    for stage in state['source_stages']:
        anchors, comparisons, semantics, limits = STAGE_REVIEW[stage['stage_id']]
        reviewed.append(dict(**stage, reviewed_semantics=semantics, limits=limits,
            numeric_nodes=STAGES[stage['stage_id']], anchors=[anchor(*a) for a in anchors],
            comparison_evidence={name:evidence['components'][name] for name in comparisons}))
    return dict(format='trackml-original-scope.v1', passed=True, approved=False, catalog_mutations=0,
        source_version_id=baseline.ORIGINAL, source_content_hash=baseline.SOURCE_HASH,
        component_revalidation_sha256=baseline.sha(baseline.ROOT/evidence_path),
        stages=reviewed,
        cell_reconstruction=dict(
            anchors=[anchor('trackml_cells', 'CellFeatures._calculateCellFeatures'),
                     anchor('decomposed_trackml_cell_lifecycle', 'execute')],
            decision='Retain the complete optional cell reconstruction as source-specific recipe logic. Keep the extracted absolute-inner-product vector selector independently reusable.',
            reason='Weighted moments feed a chosen-axis regression, pixel-extent heuristic, module thickness and rotations. This is not PCA or a general covariance estimator; replacing it with either would change the source algorithm.',
            evidence=evidence['components']['cells'],
            limits='Synthetic positive weights, complete cell information and orthonormal module rotations; no empirical detector calibration claim.'),
        reuse_decision='Twenty-five numerical callable dependencies remain individually selectable across domains under their own contracts. Detector topology, lifecycle, ranking, retry, correction-map dispatch and empirical cell policy remain in the tracking recipe. Whole-recipe portability requires those assumptions; no claim that every arithmetic expression is a separate atom.',
        compatibility_limits=[
            'Explicit empty/no-seed termination replaces source failures; sparse/exhausted layer configurations reject.',
            'Execution produces in-memory assignments; historical scoring, truth analysis and submission-file side effects are outside the execution contract.',
            'Synthetic source parity establishes the tested behavior, not historical accuracy, production scale or generalization.',
            'This automated semantic review supports Tier 3 consideration only; no human Tier 1 designation.',
            'Corrected original graph staging, stored execution, publication transactions and served retrieval remain required.'],
        reviewer_sha256=baseline.sha(__file__))


if __name__ == '__main__':
    result = review()
    (baseline.ROOT/'docs/reviews/trackml_original_scope.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, reviewed_stages=len(result['stages']), approved=False)))
