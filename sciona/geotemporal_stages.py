"""Explicit geotemporal recipe stages with reusable numeric neighborhood blending."""
from dataclasses import dataclass
import json
import math
import numpy as np

from sciona.geotemporal_contract import prepare as validate_payload, finite_json
from sciona.geotemporal_training import features as feature_matrix, points, model
from sciona.geotemporal_validation import forward_splits
from sciona.atoms.ml.neighborhoods import blend_neighbor_means


@dataclass(frozen=True)
class Prepared:
    configuration: str


@dataclass(frozen=True)
class Featured:
    prepared: Prepared
    training_features: object
    query_features: object
    folds: tuple


@dataclass(frozen=True)
class Fitted:
    featured: Featured
    estimator: object
    validation_predictions: tuple
    validation_mse: float
    fold_audits: tuple


@dataclass(frozen=True)
class Predicted:
    fitted: Fitted
    values: tuple


def prepare(payload):
    checked=validate_payload(payload).payload
    return Prepared(json.dumps(checked,sort_keys=True,allow_nan=False))


def configuration(prepared):
    if not isinstance(prepared,Prepared):raise ValueError('Prepared geotemporal configuration required')
    return validate_payload(json.loads(prepared.configuration)).payload


def features(prepared):
    p=configuration(prepared);obs=p['observations'];context=p['context'];c=p['controls']
    splits=forward_splits([r['time'] for r in obs],p['blocks'],gap=c['gap'])
    folds=[]
    for train,valid in splits:
        history=[obs[int(i)] for i in train];heldout=[obs[int(i)] for i in valid]
        folds.append(dict(train_indices=tuple(map(int,train)),validation_indices=tuple(map(int,valid)),
            training_features=feature_matrix(points(history),context,history,c),
            validation_features=feature_matrix(points(heldout),context,history,c)))
    return Featured(prepared,feature_matrix(points(obs),context,obs,c),
        feature_matrix(p['query'],context,obs,c),tuple(folds))


def postprocess(query,raw,c):
    """Recipe policy: clip, then blend other same-time sites within metric radius."""
    mask=np.array([[i!=j and a['time']==b['time'] and math.hypot(a['x']-b['x'],a['y']-b['y'])<=c['radius']
        for j,b in enumerate(query)] for i,a in enumerate(query)],dtype=bool)
    return blend_neighbor_means(np.maximum(np.asarray(raw,dtype=np.float64),0.),mask,c['smoothing']).tolist()


def fit(featured):
    if not isinstance(featured,Featured):raise ValueError('Featured geotemporal state required')
    p=configuration(featured.prepared);obs=p['observations'];c=p['controls']
    predictions=[None]*len(obs);audits=[];errors=[]
    for fold in featured.folds:
        train=fold['train_indices'];valid=fold['validation_indices']
        estimator=model(c).fit(fold['training_features'],[obs[i]['target'] for i in train])
        values=postprocess(points([obs[i] for i in valid]),estimator.predict(fold['validation_features']),c)
        for i,value in zip(valid,values,strict=True):predictions[i]=value
        errors.extend((np.asarray(values)-[obs[i]['target'] for i in valid])**2)
        audits.append(dict(fold,estimator=estimator,predictions=tuple(values)))
    estimator=model(c).fit(featured.training_features,[r['target'] for r in obs])
    mse=float(np.mean(errors))
    if not np.isfinite(mse):raise ValueError('Nonfinite held-out metric')
    return Fitted(featured,estimator,tuple(predictions),mse,tuple(audits))


def predict(fitted):
    if not isinstance(fitted,Fitted):raise ValueError('Fitted geotemporal state required')
    values=fitted.estimator.predict(fitted.featured.query_features)
    p=configuration(fitted.featured.prepared)
    if values.shape!=(len(p['query']),) or not np.isfinite(values).all():raise ValueError('Aligned finite query predictions required')
    return Predicted(fitted,tuple(map(float,values)))


def finish(predicted):
    if not isinstance(predicted,Predicted):raise ValueError('Predicted geotemporal state required')
    fitted=predicted.fitted;p=configuration(fitted.featured.prepared)
    result=dict(predictions=postprocess(p['query'],predicted.values,p['controls']),
        training_rows=len(p['observations']),query_rows=len(p['query']),
        validation_rows=sum(v is not None for v in fitted.validation_predictions),
        warmup_rows=sum(v is None for v in fitted.validation_predictions),folds=len(fitted.fold_audits),
        trees=len(fitted.estimator.estimators_),feature_count=11,validation_mse=fitted.validation_mse)
    finite_json(result)
    return result
