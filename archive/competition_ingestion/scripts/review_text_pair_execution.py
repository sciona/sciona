"""Automated Tier 3 review of the complete generic text-pair topology."""
import hashlib
import importlib.metadata as metadata
import inspect
import json
import platform
from pathlib import Path
from packaging.requirements import Requirement
ROOT=Path(__file__).resolve().parents[1]
SOURCE_VERSION='1169253d-76d3-5b76-885f-07b58bed795c'
SOURCE_HASH='896ea8e61c63f03fcb22b6bc418126ac23acf0fb3ad55e5a11c3815ae00e1bd7'
SCOPE='Generic six-stage text-pair feature ensemble topology; explicit binary matching realization, not a historical winning implementation.'
SUFFIXES=['source_triage','runtime_tests','graph_execution','environment']
LIMITATIONS=['Automated Tier 3 Community only, no Tier 1 human certification or Tier 2 usage qualification. Original generic intake remains draft with mandatory provenance.', 'Complete six-stage binary pair-matching realization using allowed lexical, LSA, graph-context, shallow ensemble and metric-threshold alternatives. No historical winning recipe or accuracy claim.', 'NFKC/casefold/word-token normalization can erase meaningful punctuation/case. Inputs limited to512 characters before normalization; empty text rejected. No external corpus, spelling correction, pretrained semantic embeddings or learned query expansion.', 'TF-IDF uses normalized unique training texts and word uni/bigrams. Truncated SVD fits only training documents; insufficient dimension rejected. Twelve symmetric feature definitions are fixed; no query-population refit.', 'Candidate graph is unlabeled co-occurrence, not known equivalence. Current training edge is excluded from its graph statistics. Empty-set Jaccard equals one; zero-norm LSA cosine equals zero.', 'Normalized unordered pairs are disjoint across training/calibration/query, with duplicate fitted pairs rejected. Individual text entities may recur across splits; no unseen-entity or temporal generalization guarantee.', 'Training-only scaler/logistic regression and random forest are equally averaged. Forest size/depth/minimum leaf and LSA size/seed are explicit. No out-of-fold stacking, probability calibration or class-balance guarantee.', 'Held-out calibration chooses maximum binary F1 with highest-threshold tie rule and score >= threshold decisions. Calibration F1 is selection evidence, not unbiased held-out accuracy. Query labels are never inputs.', 'Synthetic direct, isolation, negative-boundary and serialized lifecycle evidence only. Provisioned environment with pinned dependencies and retained installed notices; no clean-install, cross-platform, resume or resource qualification.', 'Private text and scores remain runtime values. Strict JSON input/output; intermediate fitted objects trusted in-process and no portable model serialization claim.']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def check_hashes(base,hashes):
    assert hashes
    for name,digest in hashes.items():
        path=(base/name).resolve()
        assert path.is_relative_to(base.resolve()) and path.is_file() and sha(path)==digest


def audit():
    if not __debug__:raise RuntimeError('Assertions required for review')
    reviews=ROOT/'docs/reviews'
    docs={s:json.loads((reviews/f'competition_text_pair_{s}.json').read_text()) for s in SUFFIXES}
    source=docs['source_triage'];assert source['source_version_id']==SOURCE_VERSION and source['source_content_hash']==SOURCE_HASH and source['source_scope']==SCOPE
    for suffix in SUFFIXES[1:]:assert docs[suffix]['status']=='passed'
    tests=docs['runtime_tests'];assert tests['tests_passed']==19 and tests['serialized_boundary_tests']==9
    assert tests['source_version_id']==SOURCE_VERSION and tests['source_content_hash']==SOURCE_HASH
    assert tests['stage_mapping']=={'pair_normalization': 'NFKC/casefold/word-token normalization', 'lexical_similarity_features': 'Exact match, token/character Jaccard, length ratio, edit similarity, TF-IDF cosine', 'embedding_similarity_features': 'Training-only TF-IDF and truncated SVD cosine/distance', 'pair_context_features': 'Training candidate graph, with current edge excluded for training features', 'feature_ensemble': 'Training-only scaled logistic regression plus64-tree random forest, equal score mean', 'metric_thresholding': 'Held-out binary F1 maximization with highest-threshold tie rule'}
    assert tests['checks']=={'full_feature_fit_threshold_prediction': True, 'independent_f1_search': True, 'calibration_labels_do_not_change_base_scores': True, 'query_population_independence': True, 'normalized_unordered_split_disjointness': True, 'repeated_deterministic_output': True, 'strict_json_boundary': True, 'mutated_intermediate_rejected_before_fit': True}
    check_hashes(ROOT,tests['sha256'])
    execution=docs['graph_execution']
    assert execution['checks']=={'actual_runner_nodes': 2, 'training_rows': 8, 'calibration_rows': 4, 'query_rows': 2, 'models': 2, 'forest_trees': 64, 'feature_count': 12, 'strict_json_output': True, 'graph_codec_roundtrip': True, 'provider_witness_contracts': True}
    check_hashes(ROOT,execution['code_sha256'])
    assert {str(p.relative_to(ROOT)) for p in (ROOT/'sciona').glob('text_pair_*.py')}<=set(execution['code_sha256'])
    from sciona.text_pair_graph import build_text_pair_graph
    from sciona.services.execution_graph_codec import encode_execution_graph
    import sciona.atoms.ml.text_pair_execution as provider
    graph=build_text_pair_graph()
    assert encode_execution_graph(graph)[0]==execution['serialized_graph_sha256']
    assert graph.metadata['source_version_ids']==[SOURCE_VERSION]
    witness={}
    for node in graph.nodes:
        function=getattr(provider,'text_pair_'+node.node_id)
        assert list(inspect.signature(function).parameters)==[p.name for p in node.inputs]
        assert all(p.required for p in node.inputs)
        witness=getattr(provider,'witness_text_pair_'+node.node_id)(witness)
    assert witness=={'kind':'TextPair.Result'}
    provider_path=Path(inspect.getfile(provider));assert sha(provider_path)==execution['provider_sha256']
    env=docs['environment'];requirements=ROOT/'requirements/text-pair-execution.txt'
    dependencies=dict(line.split('==') for line in requirements.read_text().splitlines())
    assert dependencies==env['dependency_versions'] and sha(requirements)==env['requirements_sha256']
    assert set(dependencies)=={'numpy','scipy','scikit-learn','joblib','threadpoolctl'}
    assert env['python_version']==platform.python_version()
    for name,version in dependencies.items():
        assert metadata.version(name)==version
        for text in metadata.requires(name) or []:
            r=Requirement(text)
            if r.marker and not r.marker.evaluate({'extra':''}):continue
            assert r.specifier.contains(metadata.version(r.name),prereleases=True)
    assert set(env['license_sha256'])=={'TextPair-numpy-3-LICENSE', 'TextPair-numpy-7-LICENSE.md', 'TextPair-numpy-10-LICENSE.md', 'TextPair-numpy-1-LICENSE.txt', 'TextPair-numpy-15-LICENSE.md', 'TextPair-numpy-0-LICENSE.txt', 'TextPair-numpy-14-LICENSE.md', 'TextPair-numpy-6-LICENSE', 'TextPair-numpy-4-dragon4_LICENSE.txt', 'TextPair-scikit-learn-0-COPYING', 'TextPair-numpy-16-LICENSE.md', 'TextPair-scipy-0-LICENSE.txt', 'TextPair-numpy-2-COPYING', 'TextPair-numpy-11-LICENSE.md', 'TextPair-numpy-13-LICENSE.md', 'TextPair-numpy-9-LICENSE', 'TextPair-numpy-12-LICENSE.md', 'TextPair-numpy-5-LICENSE.md', 'TextPair-numpy-8-LICENSE.txt'}
    check_hashes(ROOT/'docs/licenses',env['license_sha256'])
    auxiliary=['scripts/review_text_pair_execution.py','requirements/text-pair-execution.txt']+['docs/licenses/'+name for name in env['license_sha256']]
    return dict(format='text-pair-semantic-review.v1',review_source='automated',proposed_tier=3,verdict='acceptable_with_limits',
        source_version_id=SOURCE_VERSION,source_hash=SOURCE_HASH,source_scope=SCOPE,
        serialized_graph_sha256=execution['serialized_graph_sha256'],provider_sha256=sha(provider_path),
        provider_package_sha256=sha(ROOT.parent/'sciona-atoms-ml/pyproject.toml'),intake_stage_mapping=tests['stage_mapping'],
        dependencies=dependencies,limitations=LIMITATIONS,
        evidence_sha256={f'competition_text_pair_{s}.json':sha(reviews/f'competition_text_pair_{s}.json') for s in SUFFIXES},
        auxiliary_sha256={p:sha(ROOT/p) for p in auxiliary},catalog_mutations=0)


if __name__=='__main__':
    result=audit()
    (ROOT/'docs/reviews/competition_text_pair_semantic_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(verdict=result['verdict'],proposed_tier=3)))
