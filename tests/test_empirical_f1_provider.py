"""Synthetic exhaustive decisions, source parity and reusable boundary checks."""
import numpy as np
import pytest
from sciona.atoms.ml.calibration.empirical_f1 import fit_empirical_f1_thresholds, apply_column_thresholds, concrete_only
from sciona.video_multilabel_thresholds import calibrate, decisions


def brute(scores, labels):
    answers=[]
    for column, target in zip(scores.T, labels.T):
        candidates=np.r_[np.unique(column), np.nextafter(column.max(), np.inf)]
        ranked=[]
        for threshold in candidates:
            predicted=[v >= threshold for v in column]
            tp=sum(bool(p) and y == 1 for p,y in zip(predicted,target))
            fp=sum(bool(p) and y == 0 for p,y in zip(predicted,target))
            fn=sum(not p and y == 1 for p,y in zip(predicted,target))
            ranked.append((2*tp/(2*tp+fp+fn), threshold))
        answers.append(max(ranked)[1])
    return np.asarray(answers)


@pytest.mark.parametrize('seed', range(12))
def test_empirical_parity_with_duplicate_scores(seed):
    rng=np.random.default_rng(seed)
    scores=rng.integers(0,6,size=(31,4)).astype(np.float64)/5
    labels=rng.integers(0,2,size=scores.shape,dtype=np.int64)
    labels[0]=0;labels[1]=1
    original=scores.copy();original_labels=labels.copy()
    thresholds=fit_empirical_f1_thresholds(scores,labels)
    np.testing.assert_array_equal(thresholds,brute(scores,labels))
    np.testing.assert_array_equal(thresholds,calibrate(scores,labels.tolist()))
    np.testing.assert_array_equal(apply_column_thresholds(scores,thresholds),decisions(scores,thresholds.tolist()))
    np.testing.assert_array_equal(scores,original);np.testing.assert_array_equal(labels,original_labels)


def test_highest_threshold_tie_and_row_permutation():
    scores=np.array([[.9],[.8],[.7],[.6]],dtype=np.float64)
    labels=np.array([[1],[0],[0],[1]],dtype=np.int64)
    # Predicting one or all four both gives 2/3; highest threshold wins.
    np.testing.assert_array_equal(fit_empirical_f1_thresholds(scores,labels),[.9])
    np.testing.assert_array_equal(fit_empirical_f1_thresholds(scores[::-1],labels[::-1]),[.9])


@pytest.mark.parametrize('scores', [np.zeros((2,1),dtype=np.float32),np.array([[np.nan],[.5]]),np.array([[-.1],[1.]]),np.empty((0,1)),np.zeros(2)])
def test_invalid_scores(scores):
    with pytest.raises(ValueError):fit_empirical_f1_thresholds(scores,np.array([[0],[1]],dtype=np.int64))


@pytest.mark.parametrize('labels', [np.array([[False],[True]]),np.array([[0.],[1.]]),np.array([[0],[2]]),np.array([[1],[1]]),np.array([[0,1]])])
def test_invalid_labels(labels):
    with pytest.raises(ValueError):fit_empirical_f1_thresholds(np.array([[0.],[1.]]),labels)


def test_threshold_application_boundaries():
    scores=np.array([[0.,1.,.5],[1.,0.,.4]])
    np.testing.assert_array_equal(apply_column_thresholds(scores,np.array([-1.,2.,.5])),[[1,0,1],[1,0,0]])
    for thresholds in (np.zeros(2),np.array([0.,np.inf,0.]),np.zeros(3,dtype=np.float32)):
        with pytest.raises(ValueError):apply_column_thresholds(scores,thresholds)
    with pytest.raises(ValueError):concrete_only(scores)
