"""Automated Tier 3 review of the complete generic physical-operator topology."""
import hashlib
import importlib.metadata as metadata
import inspect
import json
import platform
from pathlib import Path
from packaging.requirements import Requirement
ROOT=Path(__file__).resolve().parents[1]
SOURCE_VERSION='49c645b7-3659-5cfa-a386-78f7a51f7358'
SOURCE_HASH='185e7d6e033ad27cfcf8c655d4da84ae680d6c9ec33795095b816e28e052055b'
SCOPE='Generic five-stage physics-informed operator topology; independent periodic scalar diffusion realization.'
SUFFIXES=['source_triage','runtime_tests','graph_execution','environment']
LIMITATIONS=['Automated Tier 3 Community only; no Tier 1 human certification or Tier 2 usage qualification. Original generic intake remains draft with mandatory provenance.', 'Complete five-stage periodic scalar diffusion realization: explicit physical state, learned Fourier features/operator, nonnegative mass projection and periodic smoothing. Independent project architecture, not original FNO benchmark or winning solution reproduction.', 'Uniform endpoint-excluded periodic grids only. SI length/time/diffusivity and dimensionless scalar state are explicit; nondimensional time equals diffusivity*time/length^2. Nonnegative states and mass-conserving supplied targets required.', 'Caller-supplied group identifiers must be meaningful and disjoint across training/validation/query. No automatic identity discovery; query labels rejected. Validation labels select checkpoints and cannot provide unbiased final accuracy.', 'CPU float64 full-batch Adam with learned low-mode complex Fourier multipliers, local channel mixing and GELU. Explicit width, depth, modes, epochs, learning rate and seed; earliest minimum validation-MSE checkpoint restored, including epoch zero.', 'Euclidean nonnegative initial-mass simplex projection followed by convex nearest-neighbor periodic smoothing. Uniform-grid mass and positivity preserved within numerical tolerance; this is not exact diffusion integration or enforcement of the full PDE residual.', 'Synthetic single-mode and two-mode analytic checks only. Two-mode query fixture beats its constant-mean baseline; no arbitrary PDE, out-of-distribution, resolution-transfer, uncertainty-calibration or historical accuracy guarantee.', 'Provisioned environment with pinned runtime dependency closure and installed notices. No clean-install, cross-platform, GPU, high-resolution or resource qualification. Strict JSON boundary; trusted in-process fitted objects, no portable model serialization claim.']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def check_hashes(base,hashes):
    assert hashes
    for name,digest in hashes.items():
        path=(base/name).resolve()
        assert path.is_relative_to(base.resolve()) and path.is_file() and sha(path)==digest


def audit():
    if not __debug__:raise RuntimeError('Assertions required for review')
    reviews=ROOT/'docs/reviews'
    docs={s:json.loads((reviews/f'competition_physical_operator_{s}.json').read_text()) for s in SUFFIXES}
    source=docs['source_triage'];assert source['source_version_id']==SOURCE_VERSION and source['source_content_hash']==SOURCE_HASH and source['source_scope']==SCOPE
    for suffix in SUFFIXES[1:]:assert docs[suffix]['status']=='passed'
    tests=docs['runtime_tests'];assert tests['tests_passed']==38 and tests['serialized_boundary_tests']==4
    assert tests['source_version_id']==SOURCE_VERSION and tests['source_content_hash']==SOURCE_HASH
    assert tests['stage_mapping']=={'physical_state_assembly': 'Explicit SI units, uniform periodic grid, nondimensional diffusion parameters and disjoint populations', 'spectral_or_graph_features': 'Real FFT with learned retained complex modes', 'operator_surrogate': 'Spectral/local Fourier blocks trained by CPU float64 full-batch Adam and restored best validation checkpoint', 'constraint_projection': 'Euclidean nonnegative initial-mass simplex projection', 'metric_postprocessing': 'Conservative convex nearest-neighbor periodic smoothing'}
    assert tests['checks']=={'explicit_dft_comparison': True, 'spectral_gradients': True, 'projection_kkt': True, 'validation_training_isolation': True, 'query_training_isolation': True, 'best_checkpoint_restore': True, 'mixed_mode_query_beats_mean_baseline': True, 'nonnegative_mass_conservation': True, 'strict_json_boundary': True, 'mutated_intermediate_rejected_before_fit': True}
    check_hashes(ROOT,tests['sha256'])
    execution=docs['graph_execution']
    assert execution['checks']=={'actual_runner_nodes': 2, 'training_rows': 8, 'validation_rows': 4, 'query_rows': 2, 'epochs': 60, 'grid_points': 16, 'nonnegative_mass_conservation': True, 'training_improved': True, 'strict_json_output': True, 'graph_codec_roundtrip': True, 'provider_witness_contracts': True}
    check_hashes(ROOT,execution['code_sha256'])
    assert {str(p.relative_to(ROOT)) for p in (ROOT/'sciona').glob('physical_operator_*.py')}<=set(execution['code_sha256'])
    from sciona.physical_operator_graph import build_physical_operator_graph
    from sciona.services.execution_graph_codec import encode_execution_graph
    import sciona.atoms.ml.physical_operator_execution as provider
    graph=build_physical_operator_graph()
    assert encode_execution_graph(graph)[0]==execution['serialized_graph_sha256']
    assert graph.metadata['source_version_ids']==[SOURCE_VERSION]
    witness={}
    for node in graph.nodes:
        function=getattr(provider,'physical_operator_'+node.node_id)
        assert list(inspect.signature(function).parameters)==[p.name for p in node.inputs]
        assert all(p.required for p in node.inputs)
        witness=getattr(provider,'witness_physical_operator_'+node.node_id)(witness)
    assert witness=={'kind':'PhysicalOperator.Result'}
    provider_path=Path(inspect.getfile(provider));assert sha(provider_path)==execution['provider_sha256']
    env=docs['environment'];requirements=ROOT/'requirements/physical-operator-execution.txt'
    dependencies=dict(line.split('==') for line in requirements.read_text().splitlines())
    assert dependencies==env['dependency_versions'] and sha(requirements)==env['requirements_sha256']
    assert set(dependencies)=={'jinja2', 'networkx', 'markupsafe', 'typing-extensions', 'filelock', 'torch', 'mpmath', 'setuptools', 'numpy', 'fsspec', 'sympy'}
    assert env['python_version']==platform.python_version()
    for name,version in dependencies.items():
        assert metadata.version(name)==version
        for text in metadata.requires(name) or []:
            r=Requirement(text)
            if r.marker and not r.marker.evaluate({'extra':''}):continue
            assert r.specifier.contains(metadata.version(r.name),prereleases=True)
    assert set(env['license_sha256'])=={'PhysicalOperator-setuptools-10-LICENSE.BSD', 'PhysicalOperator-fsspec-0-LICENSE', 'PhysicalOperator-setuptools-11-LICENSE', 'PhysicalOperator-setuptools-12-LICENSE', 'PhysicalOperator-networkx-0-LICENSE.txt', 'PhysicalOperator-setuptools-0-LICENSE', 'PhysicalOperator-setuptools-1-LICENSE', 'PhysicalOperator-setuptools-8-LICENSE', 'PhysicalOperator-numpy-6-LICENSE', 'PhysicalOperator-numpy-4-dragon4_LICENSE.txt', 'PhysicalOperator-setuptools-3-LICENSE', 'PhysicalOperator-setuptools-9-LICENSE.APACHE', 'PhysicalOperator-sympy-0-LICENSE', 'PhysicalOperator-jinja2-0-LICENSE.txt', 'PhysicalOperator-numpy-13-LICENSE.md', 'PhysicalOperator-setuptools-6-LICENSE', 'PhysicalOperator-numpy-1-LICENSE.txt', 'PhysicalOperator-numpy-15-LICENSE.md', 'PhysicalOperator-numpy-16-LICENSE.md', 'PhysicalOperator-setuptools-13-LICENSE.txt', 'PhysicalOperator-numpy-0-LICENSE.txt', 'PhysicalOperator-numpy-10-LICENSE.md', 'PhysicalOperator-setuptools-7-LICENSE', 'PhysicalOperator-numpy-2-COPYING', 'PhysicalOperator-numpy-3-LICENSE', 'PhysicalOperator-typing-extensions-0-LICENSE', 'PhysicalOperator-setuptools-5-LICENSE', 'PhysicalOperator-markupsafe-0-LICENSE.txt', 'PhysicalOperator-numpy-9-LICENSE', 'PhysicalOperator-numpy-7-LICENSE.md', 'PhysicalOperator-mpmath-0-LICENSE', 'PhysicalOperator-setuptools-4-LICENSE', 'PhysicalOperator-numpy-14-LICENSE.md', 'PhysicalOperator-setuptools-2-LICENSE', 'PhysicalOperator-numpy-5-LICENSE.md', 'PhysicalOperator-numpy-12-LICENSE.md', 'PhysicalOperator-filelock-0-LICENSE', 'PhysicalOperator-torch-0-LICENSE', 'PhysicalOperator-setuptools-14-LICENSE', 'PhysicalOperator-numpy-11-LICENSE.md', 'PhysicalOperator-numpy-8-LICENSE.txt'}
    check_hashes(ROOT/'docs/licenses',env['license_sha256'])
    auxiliary=['scripts/review_physical_operator_execution.py','requirements/physical-operator-execution.txt']+['docs/licenses/'+name for name in env['license_sha256']]
    return dict(format='physical-operator-semantic-review.v1',review_source='automated',proposed_tier=3,verdict='acceptable_with_limits',
        source_version_id=SOURCE_VERSION,source_hash=SOURCE_HASH,source_scope=SCOPE,
        serialized_graph_sha256=execution['serialized_graph_sha256'],provider_sha256=sha(provider_path),
        provider_package_sha256=sha(ROOT.parent/'sciona-atoms-ml/pyproject.toml'),intake_stage_mapping=tests['stage_mapping'],
        dependencies=dependencies,limitations=LIMITATIONS,
        evidence_sha256={f'competition_physical_operator_{s}.json':sha(reviews/f'competition_physical_operator_{s}.json') for s in SUFFIXES},
        auxiliary_sha256={p:sha(ROOT/p) for p in auxiliary},catalog_mutations=0)


if __name__=='__main__':
    result=audit()
    (ROOT/'docs/reviews/competition_physical_operator_semantic_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(verdict=result['verdict'],proposed_tier=3)))
