"""Read-only automated Tier 3 review of the complete Santander realization."""
import hashlib
import inspect
import json
from pathlib import Path
import platform
from scripts.validate_santander_environment import dependency_closure
ROOT=Path(__file__).resolve().parents[1]
SOURCE_VERSION='743c3225-058e-5a7e-ab18-ae5dd1c2b7d4'
SOURCE_HASH='162915709dcde12f33a88f1d693b169aa8509dbd7e5b29755f89a728d388ee11'
SCOPE='Independent complete Santander uniqueness-feature neural/tree pseudo-label ensemble on the provisioned CPU runtime.'
STAGE_MAPPING={'fake_data_identification': 'Detect singleton values within the query population alone; retain rows containing a singleton. Correct original combined-population and inverted-polarity description.', 'real_data_filtering': 'Filter query rows only for uniqueness statistics; preserve every query row and its ordering in predictions.', 'count_encoding': 'Build supervised other-row class-presence categories with self-exclusion and combined labeled/retained-query uniqueness override.', 'value_substitution': 'Replace globally unique values with the labeled-population feature mean; retain raw, categorical and substituted feature groups.', 'shuffle_augmentation': 'Preserve feature triplets within class; neural on-the-fly augmentation and tree 16-positive/four-negative shuffled copies plus originals.', 'lightgbm_training': 'Expand incomplete tree-only stage to initial shared-attention neural ranks, hard pseudo-label selection, regenerated features, ten-fold neural/tree refits and final 2.1:1 rank blend.'}
SUFFIXES=['source_review','runtime_tests','execution_boundary','default_tail_execution','graph_execution','environment']
LIMITATIONS=[
 'Automated Tier 3 Community only; no Tier 1 human review or Tier 2 usage qualification. Original intake remains draft with mandatory exact source provenance.',
 'Independent complete algorithm realization, not original winning source, weights or historical accuracy. Original source licenses were not established; no third-party script, data or model weights copied.',
 'Corrected intake: detect query-only singleton values, retain those rows for count statistics, preserve all query predictions, use supervised other-row class-presence categories and mean substitution rather than raw frequency replacement.',
 'Global labeled-reference encoding precedes folds with row self-exclusion only. Neural scaling uses training plus all query rows. CV scores are not independent generalization estimates.',
 'Shared feature-group attention network independently ported to PyTorch CPU. Explicit seeds, NumPy augmentation RNG and singleton batch merging differ from historical framework replay. Ten folds and fifteen neural epochs per stage; caller supplies seeds and other exposed controls.',
 'Initial neural ranks supply hard pseudo-label tails: neural 5000/3000 and tree 2700/2000. At least 8000 query rows required. Ambiguous equal-score boundaries reject; no silent truncation or invented tie-breaking.',
 'Selected query rows stay in the unlabeled reference while joining branch-specific labeled populations; joint stratified pseudo-label folds and reference membership are explicit independent choices. Features regenerate separately for each branch.',
 'Neural triplet shuffling occurs within class per batch. Tree augmentation retains originals plus sixteen positive/four negative shuffled copies and preserves triplets. Complete historical tree controls and categorical treatment are unresolved and exposed explicitly.',
 'Final average-tie rank blend weights neural/tree 2.1:1; output is a rank score, not a calibrated probability. Every invocation retrains the full lifecycle; no persistent serving optimization.',
 '106 synthetic tests plus actual serialized full-tail 30-model execution establish computation/boundaries on four feature groups, not historical 200-feature volume or competition strength. Provisioned CPU dependency closure and notices retained; no clean-install, GPU, cross-platform, resource-scaling or production deployment qualification.',
 'Private finite JSON inputs capped at 64 MiB; disjoint opaque identities do not detect duplicate content under renamed identities. Detailed tree control values are checked by the learner.'
]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def check_hashes(base,hashes):
    assert hashes
    for name,digest in hashes.items():
        p=(base/name).resolve()
        assert p.is_relative_to(base.resolve()) and p.is_file() and sha(p)==digest


def audit():
    if not __debug__:raise RuntimeError('Assertions required')
    reviews=ROOT/'docs/reviews';docs={s:json.loads((reviews/f'competition_santander_{s}.json').read_text()) for s in SUFFIXES}
    for d in docs.values():assert d['approved'] is False and d['catalog_mutations']==0
    source=docs['source_review'];assert source['source_scope']==SCOPE and source['classification']=='material_intake_correction_required'
    assert source['stage_mapping']==STAGE_MAPPING
    for suffix in ('source_review','runtime_tests','execution_boundary','graph_execution'):
        assert docs[suffix]['source_version_id']==SOURCE_VERSION and docs[suffix]['source_content_hash']==SOURCE_HASH
    for suffix in SUFFIXES[1:]:assert docs[suffix]['status']=='passed'
    tests=docs['runtime_tests'];assert tests['tests_passed']==106 and tests['serialized_boundary_tests']==16
    check_hashes(ROOT,tests['sha256'])
    required={str(p.relative_to(ROOT)) for p in (ROOT/'tests').glob('test_santander_*.py')}
    assert required<=set(tests['sha256'])
    boundary=docs['execution_boundary'];assert boundary['tests_passed']==16;check_hashes(ROOT,boundary['sha256'])
    defaults=docs['default_tail_execution'];assert defaults['models']==30 and defaults['folds_per_stage']==10 and defaults['neural_epochs']==15
    assert defaults['selection_counts']=={'neural':dict(positive=5000,negative=3000,labeled_rows=8060,query_reference_rows=8500),'tree':dict(positive=2700,negative=2000,labeled_rows=4760,query_reference_rows=8500)}
    check_hashes(ROOT,defaults['sha256'])
    execution=docs['graph_execution']
    assert execution['checks']==dict(actual_runner_nodes=2,models=30,synthetic_query_rows=8500,full_default_pseudo_counts=True,strict_json_output=True,graph_codec_roundtrip=True,provider_witness_contracts=True,code_unchanged_during_execution=True)
    check_hashes(ROOT,execution['code_sha256'])
    assert {str(p.relative_to(ROOT)) for p in (ROOT/'sciona').glob('santander_*.py')}<=set(execution['code_sha256'])
    from sciona.santander_graph import build_santander_graph
    from sciona.services.execution_graph_codec import encode_execution_graph
    import sciona.atoms.ml.santander_execution as provider
    graph=build_santander_graph();assert encode_execution_graph(graph)[0]==execution['serialized_graph_sha256']
    assert graph.metadata['source_version_ids']==[SOURCE_VERSION]
    witness={}
    for node in graph.nodes:
        fn=getattr(provider,'santander_'+node.node_id)
        assert list(inspect.signature(fn).parameters)==[p.name for p in node.inputs] and all(p.required for p in node.inputs)
        witness=getattr(provider,'witness_santander_'+node.node_id)(witness)
    assert witness=={'kind':'Santander.Result'}
    provider_path=Path(inspect.getfile(provider));assert sha(provider_path)==execution['provider_sha256']==boundary['provider_sha256']
    env=docs['environment'];requirements=ROOT/'requirements/santander-execution.txt'
    dependencies=dict(line.split('==') for line in requirements.read_text().splitlines())
    assert len(dependencies)==22 and dependencies==env['dependency_versions']==dependency_closure()
    assert env['requirements_sha256']==sha(requirements)
    assert env['python_version']==platform.python_version() and env['platform']==platform.system() and env['machine']==platform.machine()
    assert env['execution_device']=='cpu' and env['declared_dependency_constraints_satisfied'] is True and env['actual_root_imports_verified'] is True
    assert env['verifier_sha256']==sha(ROOT/'scripts/validate_santander_environment.py')
    assert len(env['license_sha256'])==61;check_hashes(ROOT/'docs/licenses',env['license_sha256'])
    auxiliary=['scripts/review_santander_execution.py','scripts/validate_santander_environment.py','requirements/santander-execution.txt']+['docs/licenses/'+p for p in env['license_sha256']]
    return dict(format='santander-semantic-review.v1',review_source='automated',proposed_tier=3,verdict='acceptable_with_limits',
        source_version_id=SOURCE_VERSION,source_hash=SOURCE_HASH,source_scope=SCOPE,
        serialized_graph_sha256=execution['serialized_graph_sha256'],provider_sha256=sha(provider_path),
        provider_package_sha256=sha(ROOT.parent/'sciona-atoms-ml/pyproject.toml'),intake_stage_mapping=STAGE_MAPPING,
        dependencies=dependencies,limitations=LIMITATIONS,
        evidence_sha256={f'competition_santander_{s}.json':sha(reviews/f'competition_santander_{s}.json') for s in SUFFIXES},
        auxiliary_sha256={p:sha(ROOT/p) for p in auxiliary},catalog_mutations=0)


if __name__=='__main__':
    result=audit()
    (ROOT/'docs/reviews/competition_santander_semantic_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(verdict=result['verdict'],proposed_tier=3)))
