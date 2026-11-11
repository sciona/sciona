"""Automated Tier 3 review for the corrected complete Connectomics methods."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
SOURCE_VERSION='e9f83a4e-4cb1-5297-8c04-8fd65c6e7d33'
SOURCE_HASH='c37cf5cd41c4c7f7f19af3f2f48ab3a69d94bd451822a830088abe2c632774ed'
EVIDENCE=('source_pins','source_execution','corrected_execution','graph_execution','environment')
LIMITATIONS=[
 'Automated Tier 3 Community corrected reconstruction; no Tier 1 human certification or Tier 2 community-usage qualification. Original intake remains draft.',
 'Three explicit source corrections forward scalar thresholds into sweeps and filters and filter identity into tuned weighting. This is a new execution version, not original buggy-output parity.',
 'Both complete source grids retain 120 entries, including ten duplicates and the absent 0.153 value. Weighted precision aggregation is not represented as Bayesian inference.',
 'Historical sklearn PCA is pinned and executed with sample-count variance and original whitening representation. Modern sklearn supplies validation/base interfaces only; its PCA is not substituted.',
 'Circular lowpass boundaries, filter-specific tuned activity weights, source directivity endpoint handling, strict precedence inequalities and post-aggregation normalization are preserved.',
 'Caller must explicitly choose simple/tuned and optional directivity. The latter contributes 0.003 after source normalization, with precision weight 0.997; it is not enabled implicitly.',
 'Supported inputs are finite nonnegative float32-compatible time-by-node matrices, at least three nodes and more samples than nodes. Constant channels, degenerate residual covariance and degenerate score ranges fail closed; ill-conditioned but finite estimates remain a numerical limitation.',
 'Evidence uses synthetic matrices only, including 720 independent-reference sweep fits and four actual graph executions. It does not establish historical competition accuracy, biological validity or calibrated edge probabilities.',
 'Locally provisioned hash-verified source and pinned dependencies are required. No external inputs or model downloads occur during execution; clean-environment installation and resource-cap guarantees are not claimed.',
 'Execution files explicitly declare BSD 3 clause; retained author notices and historical sklearn BSD license accompany this reconstruction. Unlicensed rank-analysis helper is not executed.',
 'Private signals and scores remain runtime values; review artifacts contain software identities and aggregate synthetic checks only.'
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    if not __debug__:raise RuntimeError('Review assertions must be enabled')
    reviews=ROOT/'docs/reviews'
    docs={s:json.loads((reviews/f'competition_connectomics_{s}.json').read_text()) for s in EVIDENCE}
    from sciona.connectomics_source import source_namespace,PINS_SHA256
    source_namespace(Path('/private/tmp/sciona_connectomics_source'))
    pins=docs['source_pins']
    assert sha(reviews/'competition_connectomics_source_pins.json')==PINS_SHA256
    assert pins['repository']=='asutera/kaggle-connectomics'
    assert pins['commit']=='b3abb178b9a9d558fdd5a7de49ae50425576a011'
    assert pins['legacy_pca']['commit']=='8d04380d474723467b5a717328efd0c9fc5bd898'
    assert pins['execution_license']['declared']=='BSD 3 clause'
    assert set(pins['execution_license']['files'])=={'code/PCA.py','code/directivity.py','code/main.py','code/utils.py'}
    cache=Path('/private/tmp/sciona_connectomics_source')
    notice=(ROOT/'docs/licenses/Connectomics-source-notices.txt').read_text()
    for name in pins['execution_license']['files']:
        header=(cache/name).read_text().split('from __future__')[0].strip()
        assert header in notice and '# License: BSD 3 clause' in header
    assert (cache/'legacy_sklearn/COPYING').read_bytes()==(ROOT/'docs/licenses/Connectomics-legacy-scikit-learn-BSD.txt').read_bytes()
    original=docs['source_execution']
    assert original['validator_sha256']==sha(ROOT/'scripts/validate_connectomics_source.py')
    assert original['source_commit']==pins['commit'] and original['legacy_sklearn_commit']==pins['legacy_pca']['commit']
    assert [(c['pca_fits'],c['distinct_preprocessed_inputs']) for c in original['sweeps']]==[(240,2),(480,4)]
    assert all(original[k] is True for k in ('threshold_argument_ignored_reproduced','legacy_precision_inverse_covariance_check','directivity_strict_bounds','finite_tuned_directivity_blend','inputs_unchanged'))
    corrected=docs['corrected_execution']
    assert corrected['validator_sha256']==sha(ROOT/'scripts/validate_connectomics_corrected.py')
    assert corrected['runtime_sha256']==sha(ROOT/'sciona/connectomics_source.py')
    assert (corrected['grid_entries'],corrected['unique_thresholds'])==(120,110)
    assert [(c['mode'],c['pca_fits']) for c in corrected['checks']]==[('simple',240),('tuned',480)]
    assert all(c['preprocessing_support_exact'] is True and c['distinct_preprocessed_inputs']>100 and
               c['max_abs_covariance_reference_error']<3e-5 and c['max_abs_preprocessing_error']<2e-6
               for c in corrected['checks'])
    graph=docs['graph_execution']
    assert graph['status']=='passed' and graph['actual_runner_nodes']==2
    assert {(c['mode'],c['directivity'],c['pca_fits'],c['exact_source_match']) for c in graph['cases']}=={
        (m,d,240 if m=='simple' else 480,True) for m in ('simple','tuned') for d in (False,True)}
    assert len(graph['cases'])==4
    assert all(graph[k] is True for k in ('independent_directivity_exact','strict_json_output','graph_codec_roundtrip',
        'provider_witness_contracts','caller_input_preserved','degenerate_covariance_rejected','tampered_source_rejected'))
    expected={str(p.relative_to(ROOT)) for p in (ROOT/'sciona').glob('connectomics_*.py')}
    assert expected <= set(graph['code_sha256'])
    for name,digest in graph['code_sha256'].items():
        path=(ROOT/name).resolve();assert path.is_relative_to(ROOT) and sha(path)==digest
    provider=ROOT.parent/'sciona-atoms-ml/src/sciona/atoms/ml/connectomics_execution.py'
    assert sha(provider)==graph['provider_sha256']
    from sciona.connectomics_graph import build_connectomics_graph
    from sciona.services.execution_graph_codec import encode_execution_graph
    assert encode_execution_graph(build_connectomics_graph())[0]==graph['serialized_graph_sha256']
    env=docs['environment']
    assert env['status']=='passed' and env['tests_passed']==16
    assert env['test_sha256']=={'tests/test_connectomics_runtime.py':sha(ROOT/'tests/test_connectomics_runtime.py')}
    dependencies=dict(line.split('==') for line in (ROOT/'requirements/connectomics-execution.txt').read_text().splitlines())
    assert dependencies==env['direct_runtime_versions']
    assert all(importlib.metadata.version(n)==v for n,v in dependencies.items())
    mapping=[('calcium_lowpass_filter','Source circular f1/f2 or f1/f2/f3/f4'),
        ('first_difference','Adjacent temporal differences'),('fluorescence_hard_threshold','Corrected scalar threshold at each grid entry'),
        ('global_activity_sample_reweighting','Simple row activity or tuned lagged activity and filter-specific powers'),
        ('pca_precision_matrix','Pinned historical PCA, 80 percent components and negative precision'),
        ('score_matrix_normalization','Diagonal minimum and min/max after weighted precision aggregation'),
        ('threshold_sweep_ensemble','Complete 120-entry weighted grid, not Bayesian inference'),
        ('temporal_precedence_directivity','Optional strict 0.2/0.5 precedence bounds with source endpoint behavior'),
        ('weighted_score_combination','Optional 0.997 precision and 0.003 precedence')]
    auxiliary=['scripts/review_connectomics_execution.py','requirements/connectomics-execution.txt',
        'docs/licenses/Connectomics-source-notices.txt','docs/licenses/Connectomics-legacy-scikit-learn-BSD.txt']
    return dict(format='connectomics-semantic-review.v1',review_source='automated',proposed_tier=3,
        verdict='acceptable_with_limits',source_version_id=SOURCE_VERSION,source_hash=SOURCE_HASH,
        source_commits={'winning_solver':pins['commit'],'historical_pca':pins['legacy_pca']['commit']},
        serialized_graph_sha256=graph['serialized_graph_sha256'],provider_sha256=sha(provider),
        provider_package_sha256=sha(ROOT.parent/'sciona-atoms-ml/pyproject.toml'),
        intake_stage_mapping=[dict(intake=s,realization=r) for s,r in mapping],
        dependencies=dependencies,limitations=LIMITATIONS,
        evidence_sha256={f'competition_connectomics_{s}.json':sha(reviews/f'competition_connectomics_{s}.json') for s in EVIDENCE},
        auxiliary_sha256={p:sha(ROOT/p) for p in auxiliary},catalog_mutations=0)


if __name__=='__main__':
    result=audit()
    (ROOT/'docs/reviews/competition_connectomics_semantic_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(verdict=result['verdict'],proposed_tier=3,intake_stages=9,catalog_mutations=0)))
