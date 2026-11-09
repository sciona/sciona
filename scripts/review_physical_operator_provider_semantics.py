"""Offline semantic contracts bound to the complete physical runtime evidence."""
import inspect
import json
import subprocess

from scripts.validate_physical_operator_complete_runtime import ROOT, sha, source_inventory
from scripts.review_video_numeric_runtime import review as runtime_review
from sciona.physical_operator_complete_graph import adapters
from sciona.constrained_array_graphs import PROVIDERS

EVIDENCE_COMMIT = '0184a7d5'
EVIDENCE = 'docs/reviews/physical_operator_complete_runtime.json'
CONTRACTS = {
    'project_nonnegative_fixed_sum': dict(
        plain='Adjust each numeric row to be nonnegative and have the supplied total, changing it as little as possible.',
        bounds='Nonempty finite real two-dimensional values and one finite nonnegative total per row. Euclidean projection onto x>=0 and sum(x)=total; fresh float64 output preserving row and column order. Zero totals yield zero rows. Fixed-sum tolerance rtol=1e-10, atol=1e-12; unrepresentable arithmetic rejected. No initial-state interpretation, units, normalization or physical meaning is inferred.',
        dependencies=[]),
    'smooth_periodic_rows': dict(
        plain='Blend each value with its two neighbors, wrapping around at the ends of each row.',
        bounds='Nonempty finite real two-dimensional values and explicit scalar weight in [0,0.5]. Circular stencil [weight,1-2*weight,weight], applied along columns. Fresh float64 output; signed inputs allowed. Preserves row sums and nonnegativity of nonnegative input up to rounding; singleton and two-column rows allowed. Periodicity is caller policy. No exact diffusion integration or arbitrary boundary handling.',
        dependencies=[]),
    'prepare': dict(
        plain='Check the physical units, periodic grid, model settings and separated groups of examples.',
        bounds='Exact finite version-1 JSON recipe. SI length/time/diffusivity, dimensionless nonnegative scalar states, uniform endpoint-excluded periodic grid of at least four points. Aligned populations; nonnegative parameters; nonnegative mass-conserving supplied training/validation targets; query targets rejected. Caller-defined groups disjoint across training, validation and query. Explicit Fourier/local model and smoothing controls. Returns detached immutable JSON configuration; does not infer group identity or physical applicability.',
        dependencies=[]),
    'fit': dict(
        plain='Train the periodic-field model and keep the earliest checkpoint with the lowest validation error.',
        bounds='PreparedOperator from qualified prepare, revalidated before training. CPU float64 full-batch Adam; learned retained complex Fourier modes plus local channel mixing and GELU. Width, depth, modes, epochs, learning rate and seed explicit. Training labels drive gradients; validation MSE selects the earliest best checkpoint including epoch zero. Query values do not train or select the model. Validation score is not independent final accuracy. Trusted in-process fitted state; forged or concurrently mutated intermediates excluded.',
        dependencies=[]),
    'predict': dict(
        plain='Use the fitted model to predict query rows without training it again.',
        bounds='Unmodified FittedOperatorStage from qualified fit. Configured query states and nondimensional diffusivity*time/length^2 feed the restored model under no-grad. Finite aligned predictions required. Preserves query order and repeated rows, with immutable tuple values. Does not fit or update weights; raw predictions need not be nonnegative or conserve mass. No arbitrary PDE, resolution-transfer or accuracy guarantee.',
        dependencies=[]),
    'project': dict(
        plain='Make each predicted row nonnegative while preserving the total of its initial state.',
        bounds='Unmodified PredictedOperatorStage from qualified predict. Compute explicit initial-state row sums and pass raw predictions plus totals to the independently reusable Euclidean projection. Uniform grid and unchanged spacing make fixed sum correspond to fixed integral. Return immutable tuple values with the same fitted context. This constraint does not enforce the full PDE residual.',
        dependencies=['project_nonnegative_fixed_sum']),
    'finish': dict(
        plain='Smooth the constrained rows and return predictions with training and conservation summaries.',
        bounds='Unmodified ProjectedOperatorStage from qualified project. Apply independently reusable periodic smoothing with the explicit recipe weight. Return finite JSON prediction rows and population/grid/epoch counts, selected validation error, initial/final training errors and maximum mean-conservation error. Input order and duplicates preserved. Validation error reflects checkpoint selection; final training error is from the last training epoch, not necessarily the restored checkpoint. No named submission format or empirical accuracy claim.',
        dependencies=['smooth_periodic_rows']),
}


def review():
    raw = (ROOT/EVIDENCE).read_bytes()
    if raw != subprocess.check_output(['git', 'show', EVIDENCE_COMMIT+':'+EVIDENCE], cwd=ROOT):
        raise ValueError('Immutable runtime evidence changed')
    evidence = json.loads(raw)
    if evidence['tests'] != dict(tests=81, failures=0, errors=0, skipped=0) or evidence['passed'] is not True:
        raise ValueError('Complete passing runtime qualification required')
    for path, digest in evidence['software_sha256'].items():
        if sha(ROOT/path) != digest:
            raise ValueError('Qualified source changed: '+path)
    if source_inventory() != evidence['providers'] or runtime_review() != evidence['runtime']:
        raise ValueError('Registered source or runtime changed')
    records = {r['runtime_fqdn'].rsplit('.', 1)[1]:r for r in evidence['providers']}
    if set(records) != set(CONTRACTS):
        raise ValueError('Every provider requires semantic review')
    functions = {f.__name__:f for f in PROVIDERS.values()}
    functions.update({name:getattr(adapters, name) for name in ('prepare','fit','predict','project','finish')})
    providers = []
    for name, r in sorted(records.items()):
        contract = CONTRACTS[name]
        signature = inspect.signature(functions[name])
        ports = [dict(direction='input', ordinal=i, name=p.name,
            type_desc=inspect.formatannotation(p.annotation), required=True, default_value_repr='',
            constraints=contract['bounds']) for i,p in enumerate(signature.parameters.values())]
        ports.append(dict(direction='output', ordinal=0, name='result',
            type_desc=inspect.formatannotation(signature.return_annotation), required=True,
            default_value_repr='', constraints=contract['bounds']))
        providers.append(dict(r, plain_description=contract['plain'], validity_bounds=contract['bounds'], ports=ports,
            dependencies=[{key:records[d][key] for key in ('runtime_fqdn','catalog_fqdn','version_id','content_hash')}
                for d in contract['dependencies']]))
    return dict(format='physical-operator-provider-semantics.v1', passed=True, approved=False, proposed_tier=3,
        catalog_mutations=0, catalog_state_checked=False, review_source='automated',
        runtime_evidence_commit=EVIDENCE_COMMIT, runtime_evidence_sha256=sha(ROOT/EVIDENCE),
        reviewer_sha256=sha(__file__), providers=providers,
        limits=['Offline semantic candidate review, not catalog approval or a human Tier 1 designation.',
            'Two numeric operations remain independently reusable; units, populations and model choices remain recipe policy.',
            '81 synthetic tests include numeric oracles, native execution and retained-recipe parity; no empirical deployment guarantee.',
            'Live catalog reconciliation, original source-scope review, transaction checks and served selection remain required.'])


if __name__ == '__main__':
    result = review()
    (ROOT/'docs/reviews/physical_operator_provider_semantics.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, providers=len(result['providers']), catalog_state_checked=False, approved=False)))
