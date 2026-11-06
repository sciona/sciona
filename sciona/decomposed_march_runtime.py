"""March feature adapters and domain-independent dense binary score fitting.

The unchanged march_* modules remain the reference. Calibration and numeric
reduction are external registered operations, never hidden calls here.
"""
import copy
import json
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sciona.march_contract import Prepared,prepare
from sciona.march_features import season_features,matchup_features

_CONTROLS={'regularization','max_iterations','tolerance','seed'}
_FEATURED={'seasons','training','calibration','prediction'}|_CONTROLS
_ARRAYS={'train_x','train_y','calibration_x','calibration_y','forward_x','reverse_x'}|_CONTROLS


def optimizer_controls(p):
    seed=p['seed'];max_iterations=p['max_iterations']
    if type(seed) is not int or not 0<=seed<2**32 or type(max_iterations) is not int or max_iterations<1:
        raise ValueError('Explicit optimizer controls required')
    if any(type(p[k]) not in (int,float) or not np.isfinite(p[k]) or p[k]<=0 for k in ('regularization','tolerance')):
        raise ValueError('Positive finite fitting controls required')


def features(prepared):
    """Preserve box aggregation, directed PageRank and ordered scaled Elo."""
    if not isinstance(prepared,Prepared):raise ValueError('Prepared March payload required')
    p=prepare(prepared.payload).payload
    optimizer_controls(p)
    seasons=season_features(p['games'],**p['feature_controls'])
    return dict(seasons=seasons,**{k:p[k] for k in _FEATURED if k!='seasons'})


def checked_featured(featured):
    if not isinstance(featured,dict) or set(featured)!=_FEATURED:
        raise ValueError('Exact feature packet fields required')
    p=copy.deepcopy(featured);optimizer_controls(p)
    seasons=p['seasons']
    if not isinstance(seasons,dict) or not seasons:raise ValueError('Nonempty season features required')
    for season,state in seasons.items():
        if type(season) is not int or season<0 or not isinstance(state,dict) or set(state)!={'teams','last_order'}:
            raise ValueError('Exact season feature state required')
        if type(state['last_order']) is not int or state['last_order']<0 or not isinstance(state['teams'],dict) or not state['teams']:
            raise ValueError('Explicit feature chronology and team vectors required')
        for team,vector in state['teams'].items():
            if type(team) is not int or team<0 or not isinstance(vector,np.ndarray) or vector.dtype!=np.float64 or vector.shape!=(5,) or not np.isfinite(vector).all():
                raise ValueError('Finite five-component team vectors required')
    return p


def pairings(featured):
    """Validate chronology and construct mirrored fitting/calibration matrices."""
    p=checked_featured(featured)
    seasons=p['seasons'];training=p['training'];calibration=p['calibration'];prediction=p['prediction']
    groups=[];identities=set()
    for rows,labeled in [(training,True),(calibration,True),(prediction,False)]:
        if not isinstance(rows,list) or not rows:raise ValueError('Nonempty explicit chronological populations required')
        features=[];reverse=[];targets=[];years=set()
        for row in rows:
            fields={'season','order','a','b'}|({'outcome'} if labeled else set())
            if not isinstance(row,dict) or set(row)!=fields or any(type(v) is not int or v<0 for v in row.values()):raise ValueError('Exact integer matchup fields required')
            year=row['season'];years.add(year)
            if year not in seasons or row['order']<=seasons[year]['last_order']:raise ValueError('Regular-season features must precede every target matchup')
            if labeled:
                key=year,row['order']
                if key in identities:raise ValueError('Duplicate labeled game')
                identities.add(key)
                if row['outcome'] not in (0,1):raise ValueError('Binary outcome required')
                targets.append(row['outcome'])
            features.append(matchup_features(seasons,year,row['a'],row['b']))
            reverse.append(matchup_features(seasons,year,row['b'],row['a']))
        groups.append((np.asarray(features),np.asarray(reverse),np.asarray(targets),years))
    train,cal,pred=groups
    if not max(train[3])<min(cal[3]) or not max(cal[3])<min(pred[3]):raise ValueError('Training, calibration and prediction seasons must be strictly separated and ordered')
    if len(training)<2 or len(calibration)<2:raise ValueError('At least two fitting and calibration matchups required')
    def augment(group):return np.concatenate(group[:2]),np.concatenate([group[2],1-group[2]])
    tx,ty=augment(train);cx,cy=augment(cal)
    return dict(train_x=tx,train_y=ty,calibration_x=cx,calibration_y=cy,
        forward_x=pred[0],reverse_x=pred[1],**{k:p[k] for k in _CONTROLS})


def checked_arrays(arrays):
    """Accept arbitrary feature widths; row separation is the caller's duty."""
    if not isinstance(arrays,dict) or set(arrays)!=_ARRAYS:raise ValueError('Exact matrix packet fields required')
    optimizer_controls(arrays)
    p={k:arrays[k] for k in _CONTROLS}
    width=None
    for name in ('train_x','calibration_x','forward_x','reverse_x'):
        x=arrays[name]
        if not isinstance(x,np.ndarray) or x.dtype!=np.float64 or x.ndim!=2 or min(x.shape)<1 or not np.isfinite(x).all():
            raise ValueError('Nonempty finite float64 feature matrices required')
        if width is not None and x.shape[1]!=width:raise ValueError('Aligned feature widths required')
        width=x.shape[1];p[name]=np.array(x,copy=True,order='C')
    if p['forward_x'].shape!=p['reverse_x'].shape:raise ValueError('Aligned prediction orientations required')
    for name,matrix in [('train_y','train_x'),('calibration_y','calibration_x')]:
        y=arrays[name]
        if (not isinstance(y,np.ndarray) or y.dtype!=np.int64 or y.ndim!=1 or y.size<2
                or y.size!=len(p[matrix]) or not np.array_equal(np.unique(y),np.array([0,1],dtype=np.int64))):
            raise ValueError('Aligned int64 labels with both binary classes required')
        p[name]=y.copy()
    return p


def fit_scores(arrays):
    """Fit only training rows; expose scores for external held-out calibration.

    Returns calibration scores, binary targets as float64, absent sample weights,
    two prediction-score vectors and counts. No season, team or box-score fields
    are consumed; compatible numeric matrices from other domains are supported.
    """
    p=checked_arrays(arrays)
    base=make_pipeline(StandardScaler(),LogisticRegression(C=p['regularization'],
        max_iter=p['max_iterations'],tol=p['tolerance'],solver='lbfgs',random_state=p['seed']))
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        base.fit(p['train_x'],p['train_y'])
    if not np.array_equal(base.classes_,[0,1]):raise ValueError('Ordered binary classes required')
    scores=[np.asarray(base.decision_function(p[k]),dtype=np.float64)
        for k in ('calibration_x','forward_x','reverse_x')]
    if any(s.ndim!=1 or not np.isfinite(s).all() for s in scores):raise ValueError('Finite binary decision scores required')
    counts=dict(training_rows=len(p['train_x']),calibration_rows=len(p['calibration_x']),predicted_matchups=len(p['forward_x']))
    return scores[0],p['calibration_y'].astype(np.float64),None,scores[1],scores[2],counts


def projection_inputs(forward,reverse):
    """Prepare fixed orientation policy for an external generic ordered sum."""
    for value in (forward,reverse):
        if (not isinstance(value,np.ndarray) or value.dtype!=np.float64 or value.ndim!=1 or not value.size
                or not np.isfinite(value).all() or np.any((value<0)|(value>1))):
            raise ValueError('Finite float64 probability vectors required')
    if forward.shape!=reverse.shape:raise ValueError('Aligned orientation predictions required')
    return np.stack([forward,np.ones_like(forward),reverse]),np.array([1.,1.,-1.])


def result(total,counts):
    """Scale the external ordered sum and restore the existing strict JSON result."""
    if (not isinstance(total,np.ndarray) or total.dtype!=np.float64 or total.ndim!=1 or not total.size
            or not np.isfinite(total).all() or np.any((total<0)|(total>2))):
        raise ValueError('Finite float64 orientation total in [0,2] required')
    if (not isinstance(counts,dict) or set(counts)!={'training_rows','calibration_rows','predicted_matchups'}
            or any(type(v) is not int for v in counts.values()) or counts['training_rows']<2
            or counts['calibration_rows']<2 or counts['predicted_matchups']!=len(total)):
        raise ValueError('Complete aligned lifecycle counts required')
    output=dict(probabilities=(.5*total).tolist(),**counts)
    json.dumps(output,allow_nan=False)
    return output
