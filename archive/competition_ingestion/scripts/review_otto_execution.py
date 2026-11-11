"""Read-only automated Tier 3 review of the complete independent Otto stack."""
import hashlib
import importlib.metadata as metadata
import inspect
import json
from pathlib import Path
from packaging.version import Version
from scripts.inventory_otto_environment import closure
ROOT=Path(__file__).resolve().parents[1]
SOURCE_VERSION='dc4969f7-e5a4-5644-a9d0-fa10493458ed'
SOURCE_HASH='13be8fd78b963daf9fd05fb5758e1d36fa98d4ed8cfdde85eb829c94f73af30a'
SCOPE='Independent implementation of the documented complete Otto stacking method, qualified on the provisioned CPU runtime.'
STAGE_MAPPING={'numerical_engineering': 'Explicit source-listed raw/log/scaled/square-root representations, fixed three-dimensional t-SNE and caller-configured clustering; no inferred GMM requirement.', 'interaction_features': 'Forest-ranked selected-feature triple products for the Sofia variants plus source-listed distance/count/cluster supplements; no invented ratio bank.', 'stratification': 'One common five-fold first-level partition and a separate four-fold meta partition; nine mutually exclusive classes, not repeated multilabel stratification.', 'base_modeling': 'All 33 enumerated source entries including native and neural families; replace generic LightGBM/MultiOutputClassifier description.', 'level_2_meta_modeling': 'XGBoost, Lasagne neural and AdaBoost-ExtraTrees selection and refits with complete 250/600/250 bags.', 'final_ensemble': 'Mean within each bag; documented geometric/arithmetic blend; normalize only after the complete formula.'}
SUFFIXES=['source_triage','full_pipeline_execution','execution_boundary','graph_execution',
          'environment_inventory','native_links','runtime_identity','native_notices']
LIMITATIONS=[
 'Automated Tier 3 Community only. No Tier 1 human review or Tier 2 usage qualification; original intake stays draft with mandatory exact provenance.',
 'Independent complete realization of the enumerated stacking recipe, not original winning code, historical weights, training volume or accuracy. Material intake corrections replace generic GMM, multilabel, LightGBM/MultiOutputClassifier and ratio descriptions.',
 'All 33 first-level entries use explicit controls and shared five-fold partitions; supervised producers exclude each held-out fold and refit on full training for queries. Identities must be disjoint; duplicated content under different identities is not detected.',
 'Three t-SNE coordinates plus two cluster features yield 293 first-level columns. Historical 297-column and supplemental-width claims remain unresolved; no invented prediction columns. Fixed population may include query features, never query labels; no out-of-sample t-SNE transform.',
 'Metric and cluster multiplicities, interaction subsets, transforms, network widths/epochs and candidate grids are caller choices. CPU implementations of the 120-model neural families preserve bag sizes but do not qualify historical GPU behavior or ambiguous layer conventions.',
 'Modern Python XGBoost/H2O interfaces, explicit SAMME, stable Sofia score conversion, assembled-block scaling and nonnegative-block log scope are independently qualified choices. Legacy implementation equivalence is unproven.',
 'Meta selection uses four folds over supplied first-level features; upstream producers are not nested inside meta folds. Selection losses are not independent generalization estimates. Raw features enter neural meta input only.',
 'Complete 250/600/250 meta bags vary seeds with selected controls fixed. Final means enter the documented geometric/arithmetic expression and normalize only afterward. Synthetic checks establish computation and boundaries, not competition strength.',
 'Provisioned macOS arm64 Python 3.13 runtime requires configured R library, libFM, Java 21 PATH and patched source-checkout Lasagne/Theano PYTHONPATH. Installed identity, recorded Mach-O links and notices verified; no clean-install, arbitrary dynamic-loading, GPU, cross-platform or resource-scaling qualification.',
 'Private finite JSON inputs are capped at 64 MiB. Preparation validates populations and control structure; detailed controls are validated by their learners. Each invocation retrains the complete stack. No production deployment or persistent-model serving claim.'
]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def check_hashes(base,hashes):
    assert hashes
    for name,digest in hashes.items():
        path=(base/name).resolve()
        assert path.is_relative_to(base.resolve()) and path.is_file() and sha(path)==digest


def audit():
    if not __debug__:raise RuntimeError('Assertions required for review')
    reviews=ROOT/'docs/reviews'
    docs={s:json.loads((reviews/f'competition_otto_{s}.json').read_text()) for s in SUFFIXES}
    for d in docs.values():assert d.get('approved') is False and d.get('catalog_mutations')==0
    for suffix in ('source_triage','full_pipeline_execution','execution_boundary','graph_execution'):
        assert docs[suffix]['source_version_id']==SOURCE_VERSION and docs[suffix]['source_content_hash']==SOURCE_HASH
    source=docs['source_triage'];assert source['source_scope']==SCOPE and source['classification']=='material_intake_correction_required'
    assert source['stage_mapping']==STAGE_MAPPING
    scope=ROOT/'docs/reviews/competition_otto_execution_scope.md'
    assert source['scope_document_sha256']==sha(scope)
    for suffix in SUFFIXES:
        if suffix not in ('source_triage','environment_inventory'):assert docs[suffix]['status']=='passed'
    full=docs['full_pipeline_execution']
    assert (full['source_entries_executed'],full['supplemental_blocks_executed'],full['first_level_columns'])==(33,7,293)
    assert (full['tree_columns'],full['neural_columns'])==(341,354) and full['gpu_validated'] is False
    assert (full['synthetic_training_rows'],full['synthetic_query_rows'],full['synthetic_query_correct'])==(1350,9,9)
    assert set(full['meta_selection'])=={'xgboost','lasagne_neural','adaboost_extratrees'}
    for family,runs in [('xgboost',250),('lasagne_neural',600),('adaboost_extratrees',250)]:
        item=full['meta_selection'][family]
        assert item['refit_models']==runs and item['tuning_model_fits']==8*runs
        assert len(item['log_losses'])==2 and item['selected_index']==1 and 0<=item['log_losses'][1]<item['log_losses'][0]
    check_hashes(ROOT,full['sha256'])
    assert full['boundary_tests_passed']==5 and full['boundary_test_sha256']==sha(ROOT/'tests/test_otto_meta_stage.py')
    assert full['scope_sha256']==sha(scope)
    boundary=docs['execution_boundary'];assert boundary['tests_passed']==26
    check_hashes(ROOT,boundary['sha256'])
    assert {'sciona/otto_execution.py','sciona/otto_graph.py','scripts/otto_synthetic.py','tests/test_otto_execution.py'}<=set(boundary['sha256'])
    execution=docs['graph_execution']
    assert execution['checks']==dict(actual_runner_nodes=2,meta_tuning_fits=8800,meta_refit_models=1100,
        synthetic_queries_correct=9,strict_json_output=True,graph_codec_roundtrip=True,
        provider_witness_contracts=True,code_unchanged_during_execution=True)
    check_hashes(ROOT,execution['code_sha256'])
    required={str(p.relative_to(ROOT)) for pattern in ('otto_*.py','otto_*.R') for p in (ROOT/'sciona').glob(pattern)}
    assert required<=set(execution['code_sha256'])
    from sciona.otto_graph import build_otto_graph
    from sciona.services.execution_graph_codec import encode_execution_graph
    import sciona.atoms.ml.otto_execution as provider
    graph=build_otto_graph()
    assert encode_execution_graph(graph)[0]==execution['serialized_graph_sha256']
    assert graph.metadata['source_version_ids']==[SOURCE_VERSION]
    witness={}
    for node in graph.nodes:
        function=getattr(provider,'otto_'+node.node_id)
        assert list(inspect.signature(function).parameters)==[p.name for p in node.inputs]
        assert all(p.required for p in node.inputs)
        witness=getattr(provider,'witness_otto_'+node.node_id)(witness)
    assert witness=={'kind':'Otto.Result'}
    provider_path=Path(inspect.getfile(provider))
    assert sha(provider_path)==execution['provider_sha256']==boundary['provider_sha256']
    inventory=docs['environment_inventory'];assert inventory['status']=='inventory_complete'
    requirements=ROOT/'requirements/otto-execution.txt'
    dependencies=dict(line.split('==') for line in requirements.read_text().splitlines())
    assert len(dependencies)==14 and dependencies==inventory['dependency_versions']==closure()
    assert sha(requirements)==inventory['requirements_sha256']
    assert inventory['declared_python_constraints_satisfied'] is True
    check_hashes(ROOT/'docs/licenses',inventory['license_sha256'])
    assert len(inventory['license_sha256'])==88 and len(inventory['source_commits'])==4
    for name,hashes in inventory['source_code_sha256'].items():check_hashes(ROOT/'.venv'/('otto-'+name),hashes)
    links=docs['native_links'];assert links['roots']==255 and len(links['native_files'])==273 and len(links['system_libraries'])==33
    assert links['inventory_sha256']==sha(reviews/'competition_otto_environment_inventory.json')
    assert links['verifier_sha256']==sha(ROOT/'scripts/validate_otto_native_links.py')
    for name,record in links['native_files'].items():assert sha(Path(name))==record['sha256']
    runtime=docs['runtime_identity'];check_hashes(ROOT,runtime['sha256'])
    assert set(runtime['python_import_versions'])==set(dependencies)
    for name,version in runtime['python_import_versions'].items():assert Version(version)==Version(dependencies[name])
    assert runtime['neural_imports']=={'lasagne':'0.2.dev1','theano':'1.0.5','device':'cpu'}
    assert runtime['r_import_versions']=={n:r['Version'] for n,r in inventory['r_packages'].items()}
    assert runtime['r_version']=='4.6.1' and runtime['java_version']=='21.0.10'
    assert (runtime['verified_native_files'],runtime['verified_notices'])==(273,88)
    notices=docs['native_notices'];check_hashes(ROOT/'docs/licenses',notices['license_sha256'])
    assert len(notices['license_sha256'])==54 and len(notices['homebrew_packages'])==18
    assert all(p['notice_origins'] for p in notices['homebrew_packages'].values())
    assert notices['native_link_evidence_sha256']==sha(reviews/'competition_otto_native_links.json')
    assert notices['verifier_sha256']==sha(ROOT/'scripts/retain_otto_native_notices.py')
    auxiliary=['scripts/review_otto_execution.py','scripts/inventory_otto_environment.py',
        'scripts/validate_otto_native_links.py','scripts/validate_otto_runtime_identity.py',
        'scripts/retain_otto_native_notices.py','requirements/otto-execution.txt',str(scope.relative_to(ROOT))]
    auxiliary+=['docs/licenses/'+n for n in set(inventory['license_sha256'])|set(notices['license_sha256'])]
    return dict(format='otto-semantic-review.v1',review_source='automated',proposed_tier=3,verdict='acceptable_with_limits',
        source_version_id=SOURCE_VERSION,source_hash=SOURCE_HASH,source_scope=SCOPE,
        serialized_graph_sha256=execution['serialized_graph_sha256'],provider_sha256=sha(provider_path),
        provider_package_sha256=sha(ROOT.parent/'sciona-atoms-ml/pyproject.toml'),intake_stage_mapping=source['stage_mapping'],
        dependencies=dependencies,limitations=LIMITATIONS,
        evidence_sha256={f'competition_otto_{s}.json':sha(reviews/f'competition_otto_{s}.json') for s in SUFFIXES},
        auxiliary_sha256={p:sha(ROOT/p) for p in auxiliary},catalog_mutations=0)


if __name__=='__main__':
    result=audit()
    (ROOT/'docs/reviews/competition_otto_semantic_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(verdict=result['verdict'],proposed_tier=3)))
