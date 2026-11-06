"""Pending original tabular OOF recipe using group-excluded training features.

Feature policy is explicit and fitted independently inside each outer fold.
The retained approved derivative is unchanged. Additional feature families and
catalog qualification remain required for the full original topology.
"""
import warnings
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from sciona.tabular_ensemble_features import validate_table
from sciona.tabular_ensemble_training import binary_labels, group_ids, BaseModels, StackedModel
from sciona.tabular_complete_features import fit_binary_tabular_features


def fit_complete_stacker(numeric, categorical, labels, folds, groups, *, feature_controls,
                         seed=42, trees=64, max_depth=6, min_leaf=1):
    """Fit all outer folds and full-training refit with one native thread.

    Base models consume the returned group-excluded training matrix, never a
    full-map transformation of their own training rows. Validation and query
    use the frozen maps. The stacker fits exactly one OOF row per training row.
    Group identifiers and fold assignment remain caller policy.
    """
    if type(feature_controls) is not dict or set(feature_controls) != {
        'clip_low', 'clip_high', 'max_mode_fraction', 'log_columns', 'interactions', 'smoothing'}:
        raise ValueError('Exact complete feature policy required')
    with threadpool_limits(limits=1):
        return _fit_complete_stacker(numeric,categorical,labels,folds,groups,
            feature_controls=feature_controls,seed=seed,trees=trees,max_depth=max_depth,min_leaf=min_leaf)


def _fit_base(numeric,categorical,labels,groups,controls):
    features,x=fit_binary_tabular_features(numeric,categorical,labels.tolist(),groups,**controls['feature_controls'])
    linear=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,random_state=controls['seed']))
    trees=ExtraTreesClassifier(n_estimators=controls['trees'],max_depth=controls['max_depth'],min_samples_leaf=controls['min_leaf'],random_state=controls['seed'],n_jobs=1)
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        linear.fit(x,labels)
    trees.fit(x,labels)
    return BaseModels(features,linear,trees)


def _fit_complete_stacker(numeric,categorical,labels,folds,groups,*,feature_controls,seed=42,trees=64,max_depth=6,min_leaf=1):
    validate_table(numeric,categorical)
    y=binary_labels(labels,len(numeric));identities=group_ids(groups,len(y))
    if not isinstance(folds,list) or len(folds)!=len(y) or any(type(f) is not int or f<0 for f in folds):raise ValueError('Invalid fold assignment')
    fold_set=set(folds)
    if len(fold_set)<2 or fold_set!=set(range(len(fold_set))):raise ValueError('Expected contiguous fold indices from zero')
    assignments={}
    for group,fold in zip(groups,folds):
        if group in assignments and assignments[group]!=fold:raise ValueError('Group crosses validation folds')
        assignments[group]=fold
    for name,value in [('seed',seed),('trees',trees),('max_depth',max_depth),('min_leaf',min_leaf)]:
        if type(value) is not int or value<(0 if name=='seed' else 1):raise ValueError('Invalid base-model control')
    if seed>=2**32:raise ValueError('Invalid random seed')
    controls=dict(seed=seed,trees=trees,max_depth=max_depth,min_leaf=min_leaf,feature_controls=feature_controls)
    folds=np.asarray(folds);splits=[]
    for fold in sorted(fold_set):
        train=np.flatnonzero(folds!=fold);valid=np.flatnonzero(folds==fold)
        if len(np.unique(y[train]))!=2:raise ValueError('Each fitting fold requires both classes')
        splits.append((fold,train,valid))
    oof=np.full((len(y),2),np.nan);counts=np.zeros(len(y),dtype=int);models={};audit=[]
    for fold,train,valid in splits:
        model=_fit_base([numeric[i] for i in train],[categorical[i] for i in train],y[train],[groups[i] for i in train],controls)
        oof[valid]=model.predict([numeric[i] for i in valid],[categorical[i] for i in valid]);counts[valid]+=1
        models[fold]=model
        audit.append(dict(fold=fold,fit_rows=len(train),validation_rows=len(valid)))
    if not np.isfinite(oof).all() or not np.all(counts==1):raise ValueError('Incomplete out-of-fold predictions')
    stacker=LogisticRegression(max_iter=1000,random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        stacker.fit(oof,y)
    full=_fit_base(numeric,categorical,y,groups,controls)
    return StackedModel(full,stacker,identities,oof,models,audit)
