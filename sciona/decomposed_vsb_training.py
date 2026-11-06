"""Explicit fold training and score boundaries for the independent VSB recipe.

Fold fitting accepts generic numeric feature matrices. Nine-column ordering,
three-signal labels and repeated five-way accumulation are explicit recipe policy.
"""
import numpy as np
import lightgbm as lgb
from .vsb_threshold import binary_labels


def recipe_inputs(training,signal_labels,query):
    x=np.asarray(training);q=np.asarray(query);labels=np.asarray(signal_labels)
    for a in (x,q):
        if a.ndim!=2 or not len(a) or a.shape[1]!=9 or a.dtype.kind not in 'iuf' or np.isinf(a).any():
            raise ValueError('Nine real features with optional NaN required')
    if labels.shape!=(len(x),3):raise ValueError('Three aligned signal labels per measurement required')
    binary_labels(labels.reshape(-1))
    order=[2,7,3,4,8,0,6,5,1]
    return x[:,order].astype(float),labels.any(axis=1).astype(int),q[:,order].astype(float),labels.copy()


def _tables(training,labels,query):
    x=np.asarray(training);q=np.asarray(query);y=binary_labels(labels)
    if x.ndim!=2 or q.ndim!=2 or not all(x.shape) or not all(q.shape) or x.shape[1]!=q.shape[1] or len(y)!=len(x):
        raise ValueError('Aligned nonempty numeric feature matrices required')
    for a in (x,q):
        if a.dtype.kind not in 'iuf' or np.isinf(a).any():raise ValueError('Real features with optional NaN required')
    return x,y,q


def _partition(plan,size,labels):
    if not isinstance(plan,(tuple,list)) or len(plan)!=3:raise ValueError('Training and two heldout index vectors required')
    indices=[]
    for value in plan:
        a=np.asarray(value)
        if a.ndim!=1 or not len(a) or a.dtype.kind not in 'iu' or np.any(a<0) or np.any(a>=size):
            raise ValueError('Nonempty in-range integer fold indices required')
        indices.append(a)
    joined=np.concatenate(indices)
    if len(joined)!=size or not np.array_equal(np.sort(joined),np.arange(size)):
        raise ValueError('Disjoint complete fold partition required')
    if len(np.unique(labels[indices[0]]))!=2:raise ValueError('Both training classes required')
    return indices


def _controls(tree_seed,rounds,patience):
    if type(tree_seed) is not int or not 0<=tree_seed<2**31:raise ValueError('Bounded tree seed required')
    if any(type(v) is not int or v<1 for v in (rounds,patience)):raise ValueError('Positive training limits required')


def fit_fold(training,labels,query,train,validation,test,tree_seed,rounds,patience):
    """Fit one binary tree model; both heldouts influence early stopping.

    Source hyperparameters and one CPU thread are fixed here. Feature count and
    row indices are generic; returned scores align with each supplied index order.
    """
    x,y,q=_tables(training,labels,query);_controls(tree_seed,rounds,patience)
    train,val,test=_partition((train,validation,test),len(x),y)
    params=dict(objective='binary',boosting='gbdt',learning_rate=.01,num_leaves=80,num_threads=1,
        metric='binary_logloss',feature_fraction=.8,bagging_freq=1,bagging_fraction=.8,seed=tree_seed,
        verbosity=-1,deterministic=True,force_col_wise=True)
    reference=lgb.Dataset(x[train],label=y[train])
    nominal_test=lgb.Dataset(x[test],label=y[test],reference=reference)
    validation=lgb.Dataset(x[val],label=y[val],reference=reference)
    model=lgb.train(params,reference,num_boost_round=rounds,
        valid_sets=[reference,nominal_test,validation],valid_names=['train','test','validation'],
        callbacks=[lgb.early_stopping(patience,first_metric_only=False,verbose=False)])
    return model.predict(x[train]),model.predict(x[val]),model.predict(x[test]),model.predict(q),model.best_iteration


def ensemble(training,labels,query,plans,repetitions,tree_seed,rounds,patience,fold_function):
    """Execute every planned model through an explicitly supplied fold callable.

    Validate complete repeated five-way coverage before fitting. Accumulation and
    divisions preserve source arithmetic order. Callable authenticity and matching
    feature provenance remain responsibilities of the qualified graph binding.
    """
    x,y,q=_tables(training,labels,query);_controls(tree_seed,rounds,patience)
    if type(repetitions) is not int or repetitions<1 or not isinstance(plans,list) or len(plans)!=5*repetitions:
        raise ValueError('Complete repeated five-fold plan required')
    if not callable(fold_function):raise ValueError('Explicit fold callable required')
    checked=[_partition(plan,len(x),y) for plan in plans]
    coverage=np.zeros((3,len(x)),dtype=np.int64)
    for plan in checked:
        for i,index in enumerate(plan):coverage[i,index]+=1
    if not np.array_equal(coverage,np.broadcast_to(np.array([3,1,1])[:,None]*repetitions,coverage.shape)):
        raise ValueError('Every row requires complete training and heldout coverage')
    train_scores=np.zeros(len(x));validation_scores=np.zeros(len(x));test_scores=np.zeros(len(x));query_scores=np.zeros(len(q));iterations=[]
    for train,val,test in checked:
        a,b,c,d,iteration=fold_function(x,y,q,train,val,test,tree_seed,rounds,patience)
        for values,size in ((a,len(train)),(b,len(val)),(c,len(test)),(d,len(q))):
            if not isinstance(values,np.ndarray) or values.shape!=(size,) or values.dtype.kind not in 'iuf' or not np.isfinite(values).all() or np.any((values<0)|(values>1)):
                raise ValueError('Aligned finite probability outputs required')
        if type(iteration) is not int or not 1<=iteration<=rounds:raise ValueError('Valid selected iteration required')
        train_scores[train]+=a;validation_scores[val]+=b;test_scores[test]+=c
        query_scores+=d/len(checked);iterations.append(iteration)
    train_scores/=3*repetitions;validation_scores/=repetitions;test_scores/=repetitions
    return dict(probabilities=query_scores,training_probabilities=train_scores,
        validation_probabilities=validation_scores,test_probabilities=test_scores,
        best_iterations=iterations,models=len(checked))


def threshold_inputs(signal_labels,training_probabilities):
    labels=np.asarray(signal_labels);scores=np.asarray(training_probabilities)
    if labels.ndim!=2 or labels.shape[1]!=3 or scores.shape!=(len(labels),):
        raise ValueError('Aligned three-signal labels and measurement scores required')
    y=binary_labels(labels.reshape(-1))
    if scores.dtype.kind not in 'iuf' or not np.isfinite(scores).all() or np.any((scores<0)|(scores>1)):
        raise ValueError('Finite measurement probabilities required')
    return y,np.repeat(scores,3)


def result(ensemble_result,selected):
    scores=np.asarray(ensemble_result['probabilities'])
    if scores.ndim!=1 or not len(scores) or not np.isfinite(scores).all() or np.any((scores<0)|(scores>1)):
        raise ValueError('Finite nonempty query probabilities required')
    if set(selected)!={'threshold','mcc'} or not np.isfinite(selected['threshold']) or not 0<=selected['threshold']<=1 or not np.isfinite(selected['mcc']):
        raise ValueError('Finite selected threshold and MCC required')
    return dict(ensemble_result,signal_decisions=np.repeat((scores>selected['threshold']).astype(int)[:,None],3,axis=1),
        threshold=selected['threshold'],threshold_fit_mcc=selected['mcc'],
        scope='Independent CPU reconstruction; both holdouts influence early stopping and final threshold uses in-training signal predictions.')
