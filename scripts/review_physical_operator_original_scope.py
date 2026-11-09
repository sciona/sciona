"""Map the exact original intake to the qualified periodic-diffusion realization."""
import hashlib
import json
import subprocess

from sciona.competition_import import FORMAT, _canonical, _id
from scripts.review_text_pair_original_scope import anchor
from scripts.review_physical_operator_provider_semantics import review as provider_review
from scripts.validate_physical_operator_complete_runtime import ROOT, sha

ORIGINAL = '49c645b7-3659-5cfa-a386-78f7a51f7358'
ORIGINAL_HASH = '185e7d6e033ad27cfcf8c655d4da84ae680d6c9ec33795095b816e28e052055b'
IDENTITY = '0b32be5f-4143-5346-97e0-75f05052add9'
STAGES = {
    'physical_state_assembly': dict(
        anchors=[('physical_operator_state', 'prepare'), ('physical_operator_state', 'tensors'),
            ('physical_operator_stages', 'prepare')],
        realization='Select a gridded periodic scalar-field realization with explicit SI length, time and diffusivity and dimensionless state. Validate uniform endpoint-excluded coordinates, finite aligned arrays, nonnegative states, supplied target mass and caller-defined separated populations. Serialize detached configuration; form nondimensional diffusivity*time/length^2.',
        limits='Caller supplies physical meaning and meaningful group identities. No automatic coordinate, unit or identity discovery. In-memory finite arrays only; chunked/HDF5 resource scaling is not qualified. Spectral, graph, telescope, ocean and other scientific input modalities are alternatives, not implemented by this realization.'),
    'spectral_or_graph_features': dict(
        anchors=[('physical_operator_network', 'SpectralConv1d.forward'), ('physical_operator_network', 'FourierOperator1d.forward')],
        realization='Select the Fourier-feature branch: real FFT along the periodic grid, retain explicit low modes, apply learned complex channel multipliers, zero unretained modes and invert with the original grid length. Broadcast nondimensional diffusion time as the second input channel; preserve sample indexing.',
        limits='Feature computation is inside each learned spectral block rather than a separately cached preprocessing artifact. No wavelet, graph, transit injection, cosmic-ray denoising, Gaussian-process noise or satellite-fusion branch is claimed. Physical indexing requires the validated uniform periodic ordering.'),
    'operator_surrogate': dict(
        anchors=[('physical_operator_network', 'FourierOperator1d'), ('physical_operator_training', 'fit'),
            ('physical_operator_stages', 'fit'), ('physical_operator_stages', 'predict')],
        realization='Select an independently implemented one-dimensional Fourier/local neural operator. Fit CPU float64 full-batch Adam on training targets with explicit architecture, optimization and seed controls. Select and restore the earliest minimum validation-MSE checkpoint including epoch zero. Predict finite query rows in original order without refitting.',
        limits='Training-regime coverage and physical applicability remain caller responsibilities. Validation labels select the checkpoint and do not provide independent final accuracy. No historical benchmark reproduction, arbitrary PDE accuracy, uncertainty head, resolution transfer, GPU or large-grid qualification. Transformer, graph, PINN/PIGNN and other model families remain alternatives.'),
    'constraint_projection': dict(
        anchors=[('physical_operator_stages', 'project'), ('constrained_array_ops', 'project_nonnegative_fixed_sum')],
        realization='Select nonnegativity and initial-mass conservation: supply initial-state row totals to an independently reusable Euclidean fixed-sum projection. The operation returns nonnegative rows with explicit sum tolerance; unrepresentable arithmetic fails. Unchanged uniform grid spacing makes row-sum conservation correspond to integral conservation.',
        limits='This enforces the selected discrete constraints, not the full PDE residual or arbitrary boundary/biogeochemical laws. The numeric provider takes explicit totals and has no intrinsic diffusion, unit or initial-state meaning.'),
    'metric_postprocessing': dict(
        anchors=[('constrained_array_ops', 'smooth_periodic_rows'), ('physical_operator_stages', 'finish')],
        realization='Select convex nearest-neighbor circular smoothing with explicit weight in [0,0.5]. Preserve nonnegativity and row sums to rounding tolerance. Return finite JSON predictions in query order, preserving duplicates, with population counts, checkpoint/training summaries and measured mean-conservation error.',
        limits='The generic scientific output is a prediction array with summaries, not a named competition file format. No weighted least-squares, uncertainty aggregation or other optional postprocessing branch is claimed. Smoothing is not an exact diffusion integrator. Final training MSE refers to the final epoch rather than necessarily the restored checkpoint.'),
}


def review():
    semantics_path = 'docs/reviews/physical_operator_provider_semantics.json'
    raw = (ROOT/semantics_path).read_bytes()
    if raw != subprocess.check_output(['git', 'show', '5eea7fe6:'+semantics_path], cwd=ROOT):
        raise ValueError('Immutable semantic review changed')
    semantics = json.loads(raw)
    if semantics != provider_review():
        raise ValueError('Qualified provider semantics changed')
    directory = ROOT.parent/'sciona-atoms/data/solution_cdgs'
    name = 'physics_informed_operator_surrogate_topology'
    snapshot = dict(format=FORMAT, template=json.loads((directory/(name+'.json')).read_text()),
        bindings=json.loads((directory/(name+'_bindings.json')).read_text()))
    digest = hashlib.sha256(_canonical(snapshot).encode()).hexdigest()
    fqdn = 'cdg.competition.'+snapshot['template']['asset_id']
    if (digest, _id(fqdn+'/'+digest), _id(fqdn)) != (ORIGINAL_HASH, ORIGINAL, IDENTITY):
        raise ValueError('Local source does not reconstruct the exact original intake')
    stages = snapshot['template']['stages']
    if len(stages) != len(STAGES) or {s['stage_id'] for s in stages} != set(STAGES):
        raise ValueError('Every original stage must be covered exactly once')
    return dict(format='physical-operator-original-scope.v1', passed=True, approved=False,
        catalog_mutations=0, catalog_state_checked=False, source_snapshot_reconstructed=True,
        source_version_id=ORIGINAL, source_content_hash=ORIGINAL_HASH, artifact_id=IDENTITY, fqdn=fqdn,
        provider_semantics_sha256=sha(ROOT/semantics_path), reviewer_sha256=sha(__file__),
        source_stage_review=[dict(source_stage={k:s[k] for k in ('stage_id','description','guarantees')},
            realization=STAGES[s['stage_id']]['realization'], limits=STAGES[s['stage_id']]['limits'],
            source_anchors=[anchor(*a) for a in STAGES[s['stage_id']]['anchors']]) for s in stages],
        reuse_decision='Two independent numeric operations with five explicit recipe adapters. Physical units, population identity, model family and conservation choices remain recipe policy.',
        remaining_requirements=['Recheck original snapshot and preserved derivative against live catalog state.',
            'Reconcile and approve exact provider versions with metadata, dependency and rollback checks.',
            'Bind, store and execute the corrected original, then verify Tier 3 publication and served retrieval.'],
        limits=['Source reconstruction proves exact intake content identity, not current database status or approval.',
            'All five source stages covered for the explicit periodic scalar-diffusion realization; optional model and modality branches are not universally implemented.',
            'Automated synthetic Tier 3 candidate evidence only; no human Tier 1 designation or empirical deployment guarantee.'])


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/physical_operator_original_scope.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, stages=len(result['source_stage_review']), exact_source=True, approved=False)))
