"""Automated Tier 3 review of the complete generic image_tabular topology."""
import hashlib
import importlib.metadata as metadata
import inspect
import json
import platform
from pathlib import Path
from packaging.requirements import Requirement
ROOT=Path(__file__).resolve().parents[1]
SOURCE_VERSION='4d7b7da1-6568-56b0-aa5e-7e2e7a275556'
SOURCE_HASH='41907e416953f825b93f7b409d3fb36f16fce58303147e7540566f90bdcc622f'
SCOPE='Generic five-stage image/tabular hybrid topology; independent handcrafted RGB and fold-ensemble MLP realization.'
SUFFIXES=['source_triage','runtime_tests','graph_execution','environment']
LIMITATIONS=['Automated Tier 3 Community only; no Tier 1 human certification or Tier 2 usage qualification. Original generic intake remains draft with mandatory provenance.', 'Complete five-stage binary image/tabular realization using allowed handcrafted visual descriptors and fused MLPs. Independent implementation; no historical winning recipe or empirical accuracy guarantee.', 'Caller supplies normalized RGB arrays in [0,1] with consistent channel meaning.31 descriptors comprise per-channel moments, four-bin histograms, horizontal/vertical absolute differences and aspect ratio. No learned visual backbone, file decoding, resizing or color-space correction.', 'Each fitting fold learns numeric median imputation, categorical vocabulary and joint feature scaling. All-missing numeric columns use zero; unknown categories become zero one-hot vectors. Missing categories use a sentinel distinct from literal text.', 'Caller-supplied groups must be meaningful, consistent within folds and disjoint across training/calibration/query. Both classes required for fitting folds and calibration. No automatic duplicate/entity discovery or chronological generalization claim.', 'CPU float64 full-batch MLP with explicit hidden size, dropout, label smoothing and fixed epochs. Initialization and dropout draws isolated within fold CPU RNG context. No early stopping or checkpoint selection; retained fold models are equally averaged, without full-data refit.', 'Complete OOF coverage means each training row predicted by its excluding fold model. OOF scores are not used to fit another model. Separate calibration chooses maximal F1 with highest-threshold tie rule. Calibration selection is not unbiased accuracy or probability calibration.', 'Synthetic numerical, isolation and serialized lifecycle evidence only. Dense features and full-batch training; large-data, clean-install, GPU and cross-platform behavior unqualified. Pinned provisioned dependencies and installed notices retained. Strict JSON boundary; trusted in-process fitted objects.']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def check_hashes(base,hashes):
    assert hashes
    for name,digest in hashes.items():
        path=(base/name).resolve()
        assert path.is_relative_to(base.resolve()) and path.is_file() and sha(path)==digest


def audit():
    if not __debug__:raise RuntimeError('Assertions required for review')
    reviews=ROOT/'docs/reviews'
    docs={s:json.loads((reviews/f'competition_image_tabular_{s}.json').read_text()) for s in SUFFIXES}
    source=docs['source_triage'];assert source['source_version_id']==SOURCE_VERSION and source['source_content_hash']==SOURCE_HASH and source['source_scope']==SCOPE
    for suffix in SUFFIXES[1:]:assert docs[suffix]['status']=='passed'
    tests=docs['runtime_tests'];assert tests['tests_passed']==31 and tests['serialized_boundary_tests']==11
    assert tests['source_version_id']==SOURCE_VERSION and tests['source_content_hash']==SOURCE_HASH
    assert tests['stage_mapping']=={'image_feature_extraction': '31 handcrafted RGB moments/histograms/gradients/aspect features', 'tabular_feature_preparation': 'Fold-local numeric median imputation and categorical one-hot encoding', 'feature_fusion': 'Concatenation and fold-local standardization of visual/metadata vectors', 'hybrid_head_training': 'CPU float64 MLP with dropout and label-smoothed BCE', 'fold_ensemble_calibration': 'Group-consistent fold models, complete OOF scores, equal fold probability mean and separate highest-threshold F1 calibration'}
    assert tests['checks']=={'hand_visual_features': True, 'training_only_metadata': True, 'hand_smoothed_loss_gradient': True, 'dropout_train_eval': True, 'fold_local_preprocessing': True, 'heldout_feature_label_isolation': True, 'complete_oof_coverage': True, 'equal_fold_ensemble': True, 'calibration_labels_do_not_change_scores': True, 'query_batch_independence': True, 'independent_f1_search': True, 'group_disjointness': True, 'strict_json_boundary': True, 'mutated_intermediate_rejected_before_fit': True}
    check_hashes(ROOT,tests['sha256'])
    execution=docs['graph_execution']
    assert execution['checks']=={'actual_runner_nodes': 2, 'training_rows': 12, 'calibration_rows': 4, 'query_rows': 2, 'folds': 3, 'oof_rows': 12, 'visual_features': 31, 'epochs': 40, 'training_improved': True, 'strict_json_output': True, 'graph_codec_roundtrip': True, 'provider_witness_contracts': True}
    check_hashes(ROOT,execution['code_sha256'])
    assert {str(p.relative_to(ROOT)) for p in (ROOT/'sciona').glob('image_tabular_*.py')}<=set(execution['code_sha256'])
    from sciona.image_tabular_graph import build_image_tabular_graph
    from sciona.services.execution_graph_codec import encode_execution_graph
    import sciona.atoms.ml.image_tabular_execution as provider
    graph=build_image_tabular_graph()
    assert encode_execution_graph(graph)[0]==execution['serialized_graph_sha256']
    assert graph.metadata['source_version_ids']==[SOURCE_VERSION]
    witness={}
    for node in graph.nodes:
        function=getattr(provider,'image_tabular_'+node.node_id)
        assert list(inspect.signature(function).parameters)==[p.name for p in node.inputs]
        assert all(p.required for p in node.inputs)
        witness=getattr(provider,'witness_image_tabular_'+node.node_id)(witness)
    assert witness=={'kind':'ImageTabular.Result'}
    provider_path=Path(inspect.getfile(provider));assert sha(provider_path)==execution['provider_sha256']
    env=docs['environment'];requirements=ROOT/'requirements/image-tabular-execution.txt'
    dependencies=dict(line.split('==') for line in requirements.read_text().splitlines())
    assert dependencies==env['dependency_versions'] and sha(requirements)==env['requirements_sha256']
    assert set(dependencies)=={'filelock', 'scipy', 'fsspec', 'numpy', 'networkx', 'sympy', 'mpmath', 'markupsafe', 'joblib', 'scikit-learn', 'jinja2', 'setuptools', 'threadpoolctl', 'torch', 'typing-extensions'}
    assert env['python_version']==platform.python_version()
    for name,version in dependencies.items():
        assert metadata.version(name)==version
        for text in metadata.requires(name) or []:
            r=Requirement(text)
            if r.marker and not r.marker.evaluate({'extra':''}):continue
            assert r.specifier.contains(metadata.version(r.name),prereleases=True)
    assert set(env['license_sha256'])=={'ImageTabular-numpy-15-LICENSE.md', 'ImageTabular-fsspec-0-LICENSE', 'ImageTabular-setuptools-6-LICENSE', 'ImageTabular-markupsafe-0-LICENSE.txt', 'ImageTabular-setuptools-12-LICENSE', 'ImageTabular-jinja2-0-LICENSE.txt', 'ImageTabular-threadpoolctl-0-LICENSE', 'ImageTabular-numpy-6-LICENSE', 'ImageTabular-numpy-1-LICENSE.txt', 'ImageTabular-setuptools-10-LICENSE.BSD', 'ImageTabular-numpy-11-LICENSE.md', 'ImageTabular-setuptools-7-LICENSE', 'ImageTabular-numpy-4-dragon4_LICENSE.txt', 'ImageTabular-numpy-16-LICENSE.md', 'ImageTabular-numpy-10-LICENSE.md', 'ImageTabular-numpy-8-LICENSE.txt', 'ImageTabular-numpy-5-LICENSE.md', 'ImageTabular-setuptools-9-LICENSE.APACHE', 'ImageTabular-setuptools-2-LICENSE', 'ImageTabular-joblib-0-LICENSE.txt', 'ImageTabular-filelock-0-LICENSE', 'ImageTabular-scipy-0-LICENSE.txt', 'ImageTabular-scikit-learn-0-COPYING', 'ImageTabular-setuptools-13-LICENSE.txt', 'ImageTabular-setuptools-8-LICENSE', 'ImageTabular-setuptools-0-LICENSE', 'ImageTabular-typing-extensions-0-LICENSE', 'ImageTabular-networkx-0-LICENSE.txt', 'ImageTabular-torch-0-LICENSE', 'ImageTabular-setuptools-3-LICENSE', 'ImageTabular-setuptools-14-LICENSE', 'ImageTabular-numpy-12-LICENSE.md', 'ImageTabular-mpmath-0-LICENSE', 'ImageTabular-numpy-13-LICENSE.md', 'ImageTabular-numpy-14-LICENSE.md', 'ImageTabular-sympy-0-LICENSE', 'ImageTabular-numpy-9-LICENSE', 'ImageTabular-setuptools-1-LICENSE', 'ImageTabular-numpy-2-COPYING', 'ImageTabular-setuptools-11-LICENSE', 'ImageTabular-numpy-3-LICENSE', 'ImageTabular-setuptools-4-LICENSE', 'ImageTabular-setuptools-5-LICENSE', 'ImageTabular-numpy-0-LICENSE.txt', 'ImageTabular-numpy-7-LICENSE.md'}
    check_hashes(ROOT/'docs/licenses',env['license_sha256'])
    auxiliary=['scripts/review_image_tabular_execution.py','requirements/image-tabular-execution.txt']+['docs/licenses/'+name for name in env['license_sha256']]
    return dict(format='image_tabular-semantic-review.v1',review_source='automated',proposed_tier=3,verdict='acceptable_with_limits',
        source_version_id=SOURCE_VERSION,source_hash=SOURCE_HASH,source_scope=SCOPE,
        serialized_graph_sha256=execution['serialized_graph_sha256'],provider_sha256=sha(provider_path),
        provider_package_sha256=sha(ROOT.parent/'sciona-atoms-ml/pyproject.toml'),intake_stage_mapping=tests['stage_mapping'],
        dependencies=dependencies,limitations=LIMITATIONS,
        evidence_sha256={f'competition_image_tabular_{s}.json':sha(reviews/f'competition_image_tabular_{s}.json') for s in SUFFIXES},
        auxiliary_sha256={p:sha(ROOT/p) for p in auxiliary},catalog_mutations=0)


if __name__=='__main__':
    result=audit()
    (ROOT/'docs/reviews/competition_image_tabular_semantic_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(verdict=result['verdict'],proposed_tier=3)))
