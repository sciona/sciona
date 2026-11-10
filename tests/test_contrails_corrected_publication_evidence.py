"""Synthetic corruption cases for the full-variant execution evidence gate."""
from copy import deepcopy
import pytest
from scripts import contrails_publication_evidence as evidence


def report(variant):
    folds = evidence.FOLDS[variant]
    return dict(stored_graph_verified_before_and_after=True, stored_version_id='synthetic-version',
        graph_sha256='a'*64, staging_plan_sha256='b'*64, passed=True, synthetic_only=True, approved=False, catalog_mutations=0,
        all_nodes_uncached=True, exact_state_routing=True, strict_json_output=True,
        source_sized_inputs_and_outputs=True, variant=variant, actual_runner_nodes=2,
        full_models_completed=len(folds), optimizer_updates=3*len(folds),
        resource_policy=deepcopy(evidence.RESOURCE_POLICY),
        folds=[dict(branch=b,fold=f,epochs=1,optimizer_updates=3,terminal_checkpoint=True) for b,f in folds],
        software_sha256={p:'0'*64 for p in ['scripts/validate_contrails_corrected_execution.py',
            'scripts/review_contrails_augmentation_runtime.py','docs/reviews/contrails_augmentation_runtime.json',
            'scripts/stage_contrails_corrected_intake.py','sciona/catalog_contrails_graph.py',
            'sciona/contrails_lifecycle.py','sciona/contrails_runtime.py']})


@pytest.mark.parametrize('variant',['v43','v47'])
def test_complete_variant(variant):
    evidence.validate_report(report(variant),variant)


@pytest.mark.parametrize('fault',['missing_fold','duplicate_fold','wrong_branch','checkpoint',
    'short_training','model_count','cached','routing','integer_pass','boolean_mutations',
    'opencv_missing','opencv_parallel','loader_workers','augmentation_missing'])
def test_reject_incomplete_or_unbounded(fault):
    value=report('v47')
    if fault=='missing_fold': value['folds'].pop()
    elif fault=='duplicate_fold': value['folds'][-1]=value['folds'][0]
    elif fault=='wrong_branch': value['folds'][0]['branch']='single'
    elif fault=='checkpoint': value['folds'][0]['terminal_checkpoint']=False
    elif fault=='short_training': value['folds'][0]['optimizer_updates']=0
    elif fault=='model_count': value['full_models_completed']=4
    elif fault=='cached': value['all_nodes_uncached']=False
    elif fault=='routing': value['exact_state_routing']=False
    elif fault=='integer_pass': value['passed']=1
    elif fault=='boolean_mutations': value['catalog_mutations']=False
    elif fault=='opencv_missing': value['resource_policy'].pop('opencv_threads')
    elif fault=='opencv_parallel': value['resource_policy']['opencv_threads']=16
    elif fault=='loader_workers': value['resource_policy']['data_loader_workers']=8
    else: value['software_sha256'].pop('docs/reviews/contrails_augmentation_runtime.json')
    with pytest.raises(ValueError): evidence.validate_report(value,'v47')


def test_four_fold_report_cannot_qualify_ten_fold_variant():
    with pytest.raises(ValueError): evidence.validate_report(report('v43'),'v47')


def test_both_stored_variants():
    evidence.validate_stored_reports({v:report(v) for v in evidence.FOLDS},
        dict(version_id='synthetic-version',graph_sha256='a'*64,plan_sha256='b'*64))


@pytest.mark.parametrize('fault', ['missing_variant','version','graph','plan','stored_flag','builder'])
def test_reject_wrong_stored_execution(fault):
    reports={v:report(v) for v in evidence.FOLDS}
    plan=dict(version_id='synthetic-version',graph_sha256='a'*64,plan_sha256='b'*64)
    if fault=='missing_variant': reports.pop('v47')
    elif fault=='version': reports['v47']['stored_version_id']='different-version'
    elif fault=='graph': reports['v47']['graph_sha256']='c'*64
    elif fault=='plan': reports['v47']['staging_plan_sha256']='d'*64
    elif fault=='stored_flag': reports['v47']['stored_graph_verified_before_and_after']=False
    else: reports['v47']['software_sha256'].pop('sciona/catalog_contrails_graph.py')
    with pytest.raises(ValueError): evidence.validate_stored_reports(reports,plan)


def test_unfinished_evidence_cannot_publish(monkeypatch):
    monkeypatch.setattr(evidence, 'EVIDENCE_COMMIT', None)
    with pytest.raises(ValueError): evidence.committed_report('unused')


def test_preserve_all_qualification_limits():
    limits=evidence.qualification_limits(
        {'limitations':['physical encoding','independent populations']},
        {'limitations':['physical encoding','CPU only']},
        {'v43':{'limits':['one epoch','random initialization']},
         'v47':{'limits':['one epoch','no accuracy claim']}})
    assert limits==['physical encoding','independent populations','CPU only',
                    'one epoch','random initialization','no accuracy claim']


@pytest.mark.parametrize('bad_limits', [[], None, [''], [False]])
def test_missing_execution_limits_rejected(bad_limits):
    with pytest.raises(ValueError):
        evidence.qualification_limits({'limitations':['source bound']},
            {'limitations':['provider bound']},
            {'v43':{'limits':['execution bound']},'v47':{'limits':bad_limits}})
