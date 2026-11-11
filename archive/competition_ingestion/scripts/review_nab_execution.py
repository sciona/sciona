"""Source-scope and computation audit for generic NAB Tier 3 execution."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
from packaging.requirements import Requirement
ROOT=Path(__file__).resolve().parents[1]
SOURCE_VERSION='e78e1ac4-49ad-5fe2-9a58-c82862d11ec7'
SOURCE_HASH='b8925fbe4d0563e5f596c38b2ef39166d47316f3e6cf704267ba4f60f9d49f7b'
COMMIT='ea702d75cc2258d9d7dd35ca8e5e2539d71f3140'
SUFFIXES=['source_pins','source_triage','scoring_probe','scoring_runtime','runtime_tests','graph_execution','environment']
LIMITATIONS=[
 'Automated Tier 3 Community generic reconstruction; no Tier 1 human certification or Tier 2 usage qualification. Original generic intake remains draft and is retained as mandatory provenance.',
 'The intake offers isolation forest OR autoencoder without naming a historical winning implementation. This realizes the isolation-forest branch with explicit new choices. It is not Numenta HTM, an autoencoder, or a historical high-tier/scoreboard-equivalent detector.',
 'One univariate ordered population is processed causally. Each full forest fits only prior trailing observation windows, then scores the current window by negative sklearn score_samples. No separate learned feature compressor or reconstruction-error model is claimed.',
 'Window length, bounded fitting history, minimum fit count, tree count, maximum samples, seed, threshold history and variance convention are explicit. Forests use all features, no bootstrap, one job, automatic contamination offset and per-step seed plus index. The offset does not affect score_samples.',
 'Three-sigma threshold statistics exclude the current anomaly score. Detection uses strict greater-than, even if the threshold exceeds one. Warmup scores/thresholds are null and flags false. Evaluation probation is separately caller-controlled; warmup anomalies can be missed and penalized.',
 'Fitting and scoring do not consume labels. Caller supplies sorted disjoint inclusive index windows for evaluation only. There is no timestamp parsing, elapsed-time weighting, resampling, missing-value imputation or independent real-world accuracy qualification.',
 'Independent NAB score reduction matches the pinned source for ordinary windows and preserves per-point confusion counts, best detection reward per window, preceding-window false-positive penalties and source probation cap semantics.',
 'Two explicit source corrections apply: thresholds below all eligible scores select the final sweep state; singleton windows use one row as the post-window penalty scale instead of dividing by zero. Exact historical parity is not claimed for undefined source behavior.',
 'Normalization uses all supplied labeled windows for the perfect score and scorable windows for the null baseline, including the source distinction when windows lie inside probation. A zero normalization denominator returns null. Scores are not clipped to zero or one hundred.',
 'All numerical evidence is generated synthetic input: prefix/future-mutation invariance, exclusion of the current observation from fitting, label-independent detection, 432 scoring comparisons and actual pipeline-runner execution. This is implementation evidence, not generalization performance.',
 'Qualified on the provisioned Python 3.13 macOS arm64 worker with pinned NumPy, SciPy and sklearn dependencies. Clean installation, other platforms, serialization of trained forests, restart/resume, resource caps and streaming transport are not qualified.',
 'NAB MIT notice retained. Only selected benchmark software was fetched; no benchmark records, anomaly label inventories, historical result files or private templates are embedded. Runtime values and results require private caller storage.'
]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def check_hashes(base,hashes):
    assert hashes
    for name,digest in hashes.items():
        path=(base/name).resolve()
        assert path.is_relative_to(base.resolve()) and path.is_file() and sha(path)==digest


def audit():
    if not __debug__:raise RuntimeError('Assertions required for semantic audit')
    reviews=ROOT/'docs/reviews';docs={s:json.loads((reviews/f'competition_nab_{s}.json').read_text()) for s in SUFFIXES}
    pins=docs['source_pins'];assert pins['repository']=='numenta/NAB' and pins['commit']==COMMIT
    identities={p['software_path']:p['sha256'] for p in pins['pins']}
    assert identities['LICENSE.txt']==sha(ROOT/'docs/licenses/NAB-MIT.txt')=='0a0b4d0b10cb1f7ed9ab2993ef93defc03447e6eba9daca1315dd32dae4877e3'
    triage=docs['source_triage'];assert triage['source_version_id']==SOURCE_VERSION and triage['source_content_hash']==SOURCE_HASH and triage['license']=='MIT'
    for suffix in SUFFIXES[2:]:assert docs[suffix]['status']=='passed'
    probe=docs['scoring_probe'];assert probe['independent_scalar_comparison_cases']==9
    assert probe['threshold_below_minimum_returns_no_score'] and probe['singleton_window_followed_by_samples_divides_by_zero']
    scoring=docs['scoring_runtime'];assert scoring['source_comparison_cases']==432 and scoring['max_absolute_score_error']<1e-12 and scoring['point_confusion_counts_exact']
    assert scoring['corrected_threshold_selection_cases']>0
    for suffix in ['scoring_probe','scoring_runtime']:
        assert docs[suffix]['source_commit']==COMMIT and docs[suffix]['source_sha256']==identities['nab/sweeper.py']
    assert probe['validator_sha256']==sha(ROOT/'scripts/validate_nab_scoring_probe.py')
    check_hashes(ROOT,scoring['hashes'])
    tests=docs['runtime_tests'];assert tests['tests_passed']==20
    assert all(tests[k] for k in ['prefix_and_future_mutation_invariance','current_point_excluded_from_fit','label_independent_detection'])
    check_hashes(ROOT,tests['hashes'])
    graph=docs['graph_execution'];assert graph['checks']==dict(actual_runner_nodes=2,models_fitted=90,stream_points=100,strict_json_output=True,graph_codec_roundtrip=True,provider_witness_contracts=True)
    assert {str(p.relative_to(ROOT)) for p in (ROOT/'sciona').glob('nab_*.py')}<=set(graph['code_sha256'])
    check_hashes(ROOT,graph['code_sha256'])
    provider=ROOT.parent/'sciona-atoms-ml/src/sciona/atoms/ml/nab_execution.py'
    assert sha(provider)==graph['provider_sha256']
    from sciona.nab_graph import build_nab_graph
    from sciona.services.execution_graph_codec import encode_execution_graph
    assert encode_execution_graph(build_nab_graph())[0]==graph['serialized_graph_sha256']
    environment=docs['environment'];requirements=ROOT/'requirements/nab-execution.txt'
    dependencies=dict(line.split('==') for line in requirements.read_text().splitlines())
    assert sha(requirements)==environment['requirements_sha256'] and dependencies==environment['direct_runtime_versions']
    assert set(dependencies)=={'numpy','scipy','scikit-learn','joblib','threadpoolctl'}
    for name,version in dependencies.items():
        assert metadata.version(name)==version
        for text in metadata.requires(name) or []:
            requirement=Requirement(text)
            if requirement.marker and not requirement.marker.evaluate({'extra':''}):continue
            assert requirement.specifier.contains(metadata.version(requirement.name),prereleases=True)
    auxiliary=['scripts/review_nab_execution.py','requirements/nab-execution.txt','docs/licenses/NAB-MIT.txt']
    return dict(format='nab-semantic-review.v1',review_source='automated',proposed_tier=3,verdict='acceptable_with_limits',
        source_version_id=SOURCE_VERSION,source_hash=SOURCE_HASH,source_commits={'benchmark':COMMIT},
        serialized_graph_sha256=graph['serialized_graph_sha256'],provider_sha256=sha(provider),provider_package_sha256=sha(ROOT.parent/'sciona-atoms-ml/pyproject.toml'),
        intake_stage_mapping=[dict(intake=a,realization=b) for a,b in [('streaming_ingestion','Ordered private numeric population processed without lookahead'),('representation_generation','Isolation forest fitted on prior trailing windows'),('predictive_scoring','Negative sklearn isolation-forest score_samples'),('dynamic_thresholding','Prior-score rolling mean plus three standard deviations'),('windowed_evaluation','Corrected NAB scoring and source null/perfect normalization')]],
        dependencies=dependencies,limitations=LIMITATIONS,evidence_sha256={f'competition_nab_{s}.json':sha(reviews/f'competition_nab_{s}.json') for s in SUFFIXES},
        auxiliary_sha256={p:sha(ROOT/p) for p in auxiliary},catalog_mutations=0)


if __name__=='__main__':
    result=audit();(ROOT/'docs/reviews/competition_nab_semantic_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(verdict=result['verdict'],proposed_tier=3,evidence=len(SUFFIXES))))
