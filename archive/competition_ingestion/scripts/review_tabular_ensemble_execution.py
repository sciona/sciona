"""Automated Tier 3 review of the complete generic tabular-ensemble topology."""
import hashlib
import importlib.metadata as metadata
import inspect
import json
import platform
from pathlib import Path
from packaging.requirements import Requirement
ROOT=Path(__file__).resolve().parents[1]
SOURCE_VERSION='9f3c8c4f-ce85-5e9d-b169-8b055562ce65'
SOURCE_HASH='3cc5a5fdfab302b5c1a37abe86916ac996f577d11b733cba0bbf7b7d4438b2cd'
SCOPE='Generic five-stage classical tabular ensemble topology; explicit binary mixed-table realization, not a historical winning implementation.'
SUFFIXES=['source_triage','runtime_tests','graph_execution','environment']
LIMITATIONS=['Automated Tier 3 Community only; no Tier 1 human certification or Tier 2 usage qualification. Original generic intake remains draft with mandatory provenance.', 'Complete five-stage binary mixed-table realization using allowed cleaning, split-aware features, shallow model zoo, OOF stacking and calibration alternatives. No historical winning recipe or accuracy claim.', 'Fold-local numeric median imputation, quantile clipping, constant and duplicate numeric removal; all-null numeric columns use zero before constant removal. Categorical one-hot and frequency encoding fit only fold training rows; unknown categories map to zeros.', 'Caller supplies meaningful group identifiers and contiguous folds. A group cannot cross folds or training/calibration/query populations. No automatic duplicate detection, group discovery or chronological validation.', 'Each fitting fold requires both classes. Scaled logistic regression and ExtraTrees produce one held-out probability per training row per base model. Logistic stacker fits this OOF matrix; final base models refit all training rows.', 'Separate regularized C=1 sigmoid fits clipped stack-score logits using calibration labels, with no base/stacker refit. Fixed0.5 decisions. No empirical calibration, improved accuracy or unbiased calibration-score guarantee.', 'Synthetic direct, isolation, negative-boundary and serialized lifecycle evidence only. Dense features; large/high-cardinality tables, resource use, cross-platform behavior and clean installation unqualified.', 'Strict finite JSON boundary; fitted intermediate objects are trusted in-process. No portable model serialization claim. Query labels are never inputs; real data is runtime-only.']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def check_hashes(base,hashes):
    assert hashes
    for name,digest in hashes.items():
        path=(base/name).resolve()
        assert path.is_relative_to(base.resolve()) and path.is_file() and sha(path)==digest


def audit():
    if not __debug__:raise RuntimeError('Assertions required for review')
    reviews=ROOT/'docs/reviews'
    docs={s:json.loads((reviews/f'competition_tabular_ensemble_{s}.json').read_text()) for s in SUFFIXES}
    source=docs['source_triage'];assert source['source_version_id']==SOURCE_VERSION and source['source_content_hash']==SOURCE_HASH and source['source_scope']==SCOPE
    for suffix in SUFFIXES[1:]:assert docs[suffix]['status']=='passed'
    tests=docs['runtime_tests'];assert tests['tests_passed']==40 and tests['serialized_boundary_tests']==15
    assert tests['source_version_id']==SOURCE_VERSION and tests['source_content_hash']==SOURCE_HASH
    assert tests['stage_mapping']=={'table_audit_cleaning': 'Fold-local median imputation, quantile clipping and constant/duplicate numeric removal', 'split_aware_feature_engineering': 'Fold-local one-hot and frequency features', 'tabular_model_zoo': 'Scaled logistic regression and ExtraTrees with64 trees per fit', 'ensemble_stacking_blending': 'Logistic stacker trained on complete OOF probabilities; full-training base refit', 'metric_calibration': 'Separate regularized sigmoid on clipped stack-score logits; fixed0.5 decision'}
    assert tests['checks']=={'complete_oof_coverage': True, 'held_out_fold_isolation': True, 'calibration_does_not_refit_stack': True, 'query_batch_independence': True, 'group_disjointness': True, 'strict_json_boundary': True, 'mutated_intermediate_rejected_before_fit': True}
    check_hashes(ROOT,tests['sha256'])
    execution=docs['graph_execution']
    assert execution['checks']=={'actual_runner_nodes': 2, 'training_rows': 12, 'calibration_rows': 4, 'query_rows': 2, 'models': 2, 'forest_trees': 64, 'folds': 3, 'oof_rows': 12, 'strict_json_output': True, 'graph_codec_roundtrip': True, 'provider_witness_contracts': True}
    check_hashes(ROOT,execution['code_sha256'])
    assert {str(p.relative_to(ROOT)) for p in (ROOT/'sciona').glob('tabular_ensemble_*.py')}<=set(execution['code_sha256'])
    from sciona.tabular_ensemble_graph import build_tabular_ensemble_graph
    from sciona.services.execution_graph_codec import encode_execution_graph
    import sciona.atoms.ml.tabular_ensemble_execution as provider
    graph=build_tabular_ensemble_graph()
    assert encode_execution_graph(graph)[0]==execution['serialized_graph_sha256']
    assert graph.metadata['source_version_ids']==[SOURCE_VERSION]
    witness={}
    for node in graph.nodes:
        function=getattr(provider,'tabular_ensemble_'+node.node_id)
        assert list(inspect.signature(function).parameters)==[p.name for p in node.inputs]
        assert all(p.required for p in node.inputs)
        witness=getattr(provider,'witness_tabular_ensemble_'+node.node_id)(witness)
    assert witness=={'kind':'TabularEnsemble.Result'}
    provider_path=Path(inspect.getfile(provider));assert sha(provider_path)==execution['provider_sha256']
    env=docs['environment'];requirements=ROOT/'requirements/tabular-ensemble-execution.txt'
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
    assert set(env['license_sha256'])=={'TabularEnsemble-numpy-2-COPYING', 'TabularEnsemble-numpy-13-LICENSE.md', 'TabularEnsemble-numpy-15-LICENSE.md', 'TabularEnsemble-numpy-14-LICENSE.md', 'TabularEnsemble-scipy-0-LICENSE.txt', 'TabularEnsemble-numpy-12-LICENSE.md', 'TabularEnsemble-numpy-5-LICENSE.md', 'TabularEnsemble-joblib-0-LICENSE.txt', 'TabularEnsemble-numpy-7-LICENSE.md', 'TabularEnsemble-numpy-11-LICENSE.md', 'TabularEnsemble-numpy-6-LICENSE', 'TabularEnsemble-numpy-0-LICENSE.txt', 'TabularEnsemble-numpy-10-LICENSE.md', 'TabularEnsemble-numpy-1-LICENSE.txt', 'TabularEnsemble-numpy-4-dragon4_LICENSE.txt', 'TabularEnsemble-numpy-8-LICENSE.txt', 'TabularEnsemble-numpy-3-LICENSE', 'TabularEnsemble-scikit-learn-0-COPYING', 'TabularEnsemble-numpy-16-LICENSE.md', 'TabularEnsemble-threadpoolctl-0-LICENSE', 'TabularEnsemble-numpy-9-LICENSE'}
    check_hashes(ROOT/'docs/licenses',env['license_sha256'])
    auxiliary=['scripts/review_tabular_ensemble_execution.py','requirements/tabular-ensemble-execution.txt']+['docs/licenses/'+name for name in env['license_sha256']]
    return dict(format='tabular-ensemble-semantic-review.v1',review_source='automated',proposed_tier=3,verdict='acceptable_with_limits',
        source_version_id=SOURCE_VERSION,source_hash=SOURCE_HASH,source_scope=SCOPE,
        serialized_graph_sha256=execution['serialized_graph_sha256'],provider_sha256=sha(provider_path),
        provider_package_sha256=sha(ROOT.parent/'sciona-atoms-ml/pyproject.toml'),intake_stage_mapping=tests['stage_mapping'],
        dependencies=dependencies,limitations=LIMITATIONS,
        evidence_sha256={f'competition_tabular_ensemble_{s}.json':sha(reviews/f'competition_tabular_ensemble_{s}.json') for s in SUFFIXES},
        auxiliary_sha256={p:sha(ROOT/p) for p in auxiliary},catalog_mutations=0)


if __name__=='__main__':
    result=audit()
    (ROOT/'docs/reviews/competition_tabular_ensemble_semantic_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(verdict=result['verdict'],proposed_tier=3)))
