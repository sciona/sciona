"""Automated Tier 3 review of the complete generic santa topology."""
import hashlib
import importlib.metadata as metadata
import inspect
import json
import platform
from pathlib import Path
from packaging.requirements import Requirement
ROOT=Path(__file__).resolve().parents[1]
SOURCE_VERSION='8ab2dc68-65da-5415-bf3a-bbebde6fe898'
SOURCE_HASH='97673ed4eca3bc8d89970aa96b364c4324f0b14b711c45f88c0b282831488291'
SCOPE='Source-corrected Santa 2020 winner algorithm; independent replay-trained two-LightGBM threshold agent.'
SUFFIXES=['source_triage','runtime_tests','graph_execution','environment']
LIMITATIONS=['Automated Tier 3 Community only; no Tier 1 human certification or Tier 2 usage qualification. Original intake remains draft with mandatory source provenance.', 'Original five-node Beta/UCB intake is materially corrected using the official winner index, winner explanation and linked implementation: discrete posterior and two LightGBM threshold predictors replace Beta updates and UCB. Independent implementation, not original code or model weights.', 'Fixed 100 arms, initial integer thresholds 0..100, shared .97 decay and 2000-round horizon. Own rewards update normalized log likelihoods; opponent rewards and hidden thresholds are excluded from query input. Posterior exponent decay follows source; simulator uses iterative threshold decay.', 'Ten float32 causal features use global consecutive-opponent count, distinct/max selection counts, normalized sorted-count statistic, own count and posterior mean, opponent count/inverse-decay count and time since last selection. The source statistic called Gini is not conventional Gini.', 'Caller provides valid private replays and eligible perspectives, with disjoint game identities for training, validation and query. Hidden initial thresholds enter replay labels only. Identity separation does not detect duplicated content under different identifiers. Bounded in-memory input and sampled rows; no historical replay ingestion.', 'Independent seeded one-in-five row sampling replaces source xorshift membership; separate population RNG streams and explicit game splits. Historical training population and sampling membership not reproduced.', 'Two RMSE GBDTs target initial threshold and 1.02 raised to threshold. Source defaults 4095 leaves, .05 rate, .9 feature fraction, .5 bagging every five rounds, 1024-round cap and patience 50. Held-out early stopping selects minimum RMSE. Small synthetic populations qualify these controls, not historical volume.', 'Predicted exponential targets are inverted then blended with raw predictions using normalized sigmoid schedule; shared decay follows. Explicit seeded tie order replaces wall-clock random seed; all nonpositive scores retain source action-zero fallback. Strict query replay-to-action boundary refits from input replays per invocation; persistent model-serving optimization unqualified.', '62 synthetic tests, serialized graph and full 2000-round games against a random opponent establish computation/isolation only, not winner strength. Pinned CPU single-thread installed environment and notices retained; clean-install, cross-platform, throughput and competitive accuracy unqualified.']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def check_hashes(base,hashes):
    assert hashes
    for name,digest in hashes.items():
        path=(base/name).resolve()
        assert path.is_relative_to(base.resolve()) and path.is_file() and sha(path)==digest


def audit():
    if not __debug__:raise RuntimeError('Assertions required for review')
    reviews=ROOT/'docs/reviews'
    docs={s:json.loads((reviews/f'competition_santa_{s}.json').read_text()) for s in SUFFIXES}
    source=docs['source_triage'];assert source['source_version_id']==SOURCE_VERSION and source['source_content_hash']==SOURCE_HASH and source['source_scope']==SCOPE
    assert source['classification']=='material_intake_correction_required' and source['stage_mapping']=={'belief_state_initialization': 'Correct Beta prior to uniform discrete initial thresholds 0..100', 'belief_updating': 'Own-reward discrete likelihood under shared selection decay', 'opponent_modeling': 'Ten causal features with opponent selection history and own posterior means', 'bayesian_ucb_sampling': 'Replace UCB with raw/exponential-target LightGBM training, held-out early stopping and decayed sigmoid prediction blend', 'action_execution': 'Select maximal blended decayed score using explicit seeded randomized ties'}
    for suffix in SUFFIXES[1:]:assert docs[suffix]['status']=='passed'
    tests=docs['runtime_tests'];assert tests['tests_passed']==62 and tests['serialized_boundary_tests']==11
    assert tests['source_version_id']==SOURCE_VERSION and tests['source_content_hash']==SOURCE_HASH
    assert tests['stage_mapping']=={'belief_state_initialization': 'Correct Beta prior to uniform discrete initial thresholds 0..100', 'belief_updating': 'Own-reward discrete likelihood under shared selection decay', 'opponent_modeling': 'Ten causal features with opponent selection history and own posterior means', 'bayesian_ucb_sampling': 'Replace UCB with raw/exponential-target LightGBM training, held-out early stopping and decayed sigmoid prediction blend', 'action_execution': 'Select maximal blended decayed score using explicit seeded randomized ties'}
    assert tests['checks']=={'discrete_posterior': True, 'causal_pre_round_features': True, 'opponent_reward_isolation': True, 'hidden_threshold_label_only': True, 'disjoint_games': True, 'independent_rmse': True, 'early_stopping_triggered': True, 'selected_iteration_predictions': True, 'full_2000_round_games': True, 'strict_boundary': True, 'query_model_selection_isolation': True}
    check_hashes(ROOT,tests['sha256'])
    required_tests={str(p.relative_to(ROOT)) for p in (ROOT/'tests').glob('test_santa_*.py')}
    assert required_tests <= set(tests['sha256'])
    defaults=json.loads((reviews/'competition_santa2020_default_training.json').read_text())
    assert defaults['status']=='passed' and defaults['controls']==dict(num_leaves=4095,num_boost_round=1024,stopping_rounds=50)
    assert defaults['checks']==dict(default_source_controls=True,selected_minimum_heldout_rmse=True,full_game_rounds=2000)
    assert defaults['best_iterations']=={'normal':12,'transformed':12}
    assert defaults['evaluated_iterations']=={'normal':62,'transformed':62}
    check_hashes(ROOT,defaults['sha256'])

    execution=docs['graph_execution']
    assert execution['checks']=={'actual_runner_nodes': 2, 'training_rows': 3142, 'validation_rows': 3186, 'models': 2, 'query_step': 1, 'action_count': 100, 'finite_scores': True, 'heldout_best_iterations': True, 'strict_json_output': True, 'graph_codec_roundtrip': True, 'provider_witness_contracts': True}
    check_hashes(ROOT,execution['code_sha256'])
    assert {str(p.relative_to(ROOT)) for p in (ROOT/'sciona').glob('santa_*.py')}<=set(execution['code_sha256'])
    from sciona.santa_graph import build_santa_graph
    from sciona.services.execution_graph_codec import encode_execution_graph
    import sciona.atoms.ml.santa_execution as provider
    graph=build_santa_graph()
    assert encode_execution_graph(graph)[0]==execution['serialized_graph_sha256']
    assert graph.metadata['source_version_ids']==[SOURCE_VERSION]
    witness={}
    for node in graph.nodes:
        function=getattr(provider,'santa_'+node.node_id)
        assert list(inspect.signature(function).parameters)==[p.name for p in node.inputs]
        assert all(p.required for p in node.inputs)
        witness=getattr(provider,'witness_santa_'+node.node_id)(witness)
    assert witness=={'kind':'Santa.Result'}
    provider_path=Path(inspect.getfile(provider));assert sha(provider_path)==execution['provider_sha256']
    env=docs['environment'];requirements=ROOT/'requirements/santa-execution.txt'
    dependencies=dict(line.split('==') for line in requirements.read_text().splitlines())
    assert dependencies==env['dependency_versions'] and sha(requirements)==env['requirements_sha256']
    assert set(dependencies)=={'numpy', 'python-dateutil', 'tzdata', 'scikit-learn', 'pandas', 'scipy', 'pytz', 'six', 'narwhals', 'joblib', 'lightgbm', 'threadpoolctl'}
    assert env['python_version']==platform.python_version()
    for name,version in dependencies.items():
        assert metadata.version(name)==version
        for text in metadata.requires(name) or []:
            r=Requirement(text)
            if r.marker and not r.marker.evaluate({'extra':''}):continue
            assert r.specifier.contains(metadata.version(r.name),prereleases=True)
    assert set(env['license_sha256'])=={'Santa-numpy-10-LICENSE.md', 'Santa-pandas-0-LICENSE', 'Santa-numpy-16-LICENSE.md', 'Santa-pytz-0-LICENSE.txt', 'Santa-threadpoolctl-0-LICENSE', 'Santa-numpy-14-LICENSE.md', 'Santa-numpy-7-LICENSE.md', 'Santa-python-dateutil-0-LICENSE', 'Santa-numpy-6-LICENSE', 'Santa-numpy-5-LICENSE.md', 'Santa-numpy-11-LICENSE.md', 'Santa-six-0-LICENSE', 'Santa-scipy-0-LICENSE.txt', 'Santa-narwhals-0-LICENSE.md', 'Santa-scikit-learn-0-COPYING', 'Santa-numpy-2-COPYING', 'Santa-numpy-4-dragon4_LICENSE.txt', 'Santa-numpy-0-LICENSE.txt', 'Santa-numpy-9-LICENSE', 'Santa-numpy-12-LICENSE.md', 'Santa-tzdata-0-LICENSE', 'Santa-numpy-8-LICENSE.txt', 'Santa-numpy-1-LICENSE.txt', 'Santa-tzdata-1-LICENSE_APACHE', 'Santa-numpy-13-LICENSE.md', 'Santa-lightgbm-0-LICENSE', 'Santa-numpy-15-LICENSE.md', 'Santa-joblib-0-LICENSE.txt', 'Santa-numpy-3-LICENSE'}
    check_hashes(ROOT/'docs/licenses',env['license_sha256'])
    auxiliary=['scripts/review_santa_execution.py','requirements/santa-execution.txt','docs/reviews/competition_santa2020_default_training.json']+['docs/licenses/'+name for name in env['license_sha256']]
    return dict(format='santa-semantic-review.v1',review_source='automated',proposed_tier=3,verdict='acceptable_with_limits',
        source_version_id=SOURCE_VERSION,source_hash=SOURCE_HASH,source_scope=SCOPE,
        serialized_graph_sha256=execution['serialized_graph_sha256'],provider_sha256=sha(provider_path),
        provider_package_sha256=sha(ROOT.parent/'sciona-atoms-ml/pyproject.toml'),intake_stage_mapping=tests['stage_mapping'],
        dependencies=dependencies,limitations=LIMITATIONS,
        evidence_sha256={f'competition_santa_{s}.json':sha(reviews/f'competition_santa_{s}.json') for s in SUFFIXES},
        auxiliary_sha256={p:sha(ROOT/p) for p in auxiliary},catalog_mutations=0)


if __name__=='__main__':
    result=audit()
    (ROOT/'docs/reviews/competition_santa_semantic_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(verdict=result['verdict'],proposed_tier=3)))
