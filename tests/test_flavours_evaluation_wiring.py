"""Evaluation routing, isolation, strict thresholds and reused metric parity."""
import numpy as np
import pytest
from sciona.flavours_evaluation import evaluate
from sciona.atoms.ml.flavours_evaluation_wiring import evaluation_inputs,split_population_scores,discrimination_inputs,diagnostic_report,witness_diagnostic_report
from sciona.ghost.abstract import AbstractArray,AbstractScalar
from sciona.atoms.ml.constrained_ml.decorrelation.atoms import compute_ks_agreement,compute_cvm_mass_decorrelation,roc_auc_truncated_weighted


def physical(rows):
    return [np.ones((rows,46)),np.full((rows,3),2.),np.ones((rows,3)),np.ones(rows),np.ones(rows),np.ones(rows)]


def controls():
    return dict(agreement_a=physical(7),agreement_b=physical(9),correlation=physical(300),
        weights_a=np.ones(7),weights_b=np.ones(9),correlation_mass=np.arange(300,dtype=float),quality=np.ones(10))


def test_control_metadata_changes_never_change_merged_physical_features():
    labels=np.arange(10,dtype=np.int64)%2;c=controls();first=evaluation_inputs(physical(5),c,labels)
    c.update(weights_a=np.arange(1,8,dtype=float),weights_b=np.arange(1,10,dtype=float),
             correlation_mass=np.arange(300,dtype=float)[::-1],quality=np.full(10,.5))
    second=evaluation_inputs(physical(5),c,labels)
    assert first[1]==second[1]==[5,7,9,300]
    for a,b in zip(first[0],second[0]):np.testing.assert_array_equal(a,b)
    scores=np.arange(321,dtype=float)/321
    pieces=split_population_scores(scores,first[1])
    np.testing.assert_array_equal(np.concatenate(pieces),scores)


def test_metrics_and_failed_diagnostics_match_retained_evaluator():
    labels=np.arange(10,dtype=np.int64)%2;quality=np.ones(10);quality[:2]=.4
    oof=np.linspace(.1,.9,10);a=np.zeros(7);b=np.ones(9);corr=np.linspace(0,1,300);mass=np.arange(300,dtype=float)
    expected=evaluate(labels,oof,quality,a,b,np.ones(7),np.ones(9),corr,mass)
    selected_labels,selected_scores=discrimination_inputs(labels,oof,quality)
    np.testing.assert_array_equal(selected_labels,labels[2:])
    np.testing.assert_array_equal(selected_scores,oof[2:])
    actual=diagnostic_report(np.array([.2]),compute_ks_agreement(a,b,np.ones(7),np.ones(9)),
        compute_cvm_mass_decorrelation(corr,mass,200,50),roc_auc_truncated_weighted(selected_labels,selected_scores),[1,7,9,300])
    assert actual['evaluation']==expected
    assert not actual['evaluation']['agreement_passed'] and not actual['evaluation']['correlation_passed']
    boundary=diagnostic_report(np.array([.2]),.09,.002,.5,[1,7,9,300])
    assert not boundary['evaluation']['agreement_passed'] and not boundary['evaluation']['correlation_passed']


def test_report_witness_matches_runtime_result_fields():
    metadata=witness_diagnostic_report(AbstractArray(shape=(1,)),AbstractScalar(dtype='float64'),
        AbstractScalar(dtype='float64'),AbstractScalar(dtype='float64'),[1,7,9,300])
    actual=diagnostic_report(np.array([.2]),.1,.003,.5,[1,7,9,300])
    assert set(metadata)==set(actual) and set(metadata['evaluation'])==set(actual['evaluation'])
    assert metadata['population_sizes']==actual['population_sizes']
    assert metadata['evaluation']['agreement_passed'].dtype=='bool'


@pytest.mark.parametrize('fault',['missing','weight_shape','negative_weights','zero_weights','quality_class','short_correlation','invalid_physical'])
def test_invalid_evaluation_controls_fail_before_training(fault):
    c=controls();labels=np.arange(10,dtype=np.int64)%2
    if fault=='missing':del c['quality']
    elif fault=='weight_shape':c['weights_a']=np.ones(8)
    elif fault=='negative_weights':c['weights_a'][0]=-1
    elif fault=='zero_weights':c['weights_a'][:]=0
    elif fault=='quality_class':c['quality']=labels.astype(float)
    elif fault=='short_correlation':c['correlation']=physical(199);c['correlation_mass']=np.ones(199)
    else:c['agreement_a'][4][0]=0
    with pytest.raises(ValueError):evaluation_inputs(physical(5),c,labels)
