"""Synthetic full-lifecycle equivalence and isolation under sharded access."""
import json
import numpy as np
import pytest
import torch
from scripts.video_multilabel_synthetic import payload
from sciona.video_multilabel_contract import prepare, execute as reference_execute
from sciona.video_multilabel_training import fit as reference_fit
from sciona.video_multilabel_shards import FeatureShards
from sciona.video_multilabel_sharded_training import fit, execute


def store(tmp_path, name, part):
    rows=[]
    for i, frames in enumerate(part['videos']):
        row=dict(frames=frames, sparse=part['sparse'][i], group=part['groups'][i])
        if 'labels' in part:row['labels']=part['labels'][i]
        rows.append(row)
    paths=[]
    for n,start in enumerate(range(0,len(rows),3)):
        path=tmp_path/f'{name}-{n}.jsonl'
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows[start:start+3]))
        paths.append(path)
    return FeatureShards(paths,labelled='labels' in part)


@pytest.mark.parametrize('seed,epochs,batch_size', [(12,80,4),(3,5,3),(29,3,16)])
def test_complete_lifecycle_matches_reference(tmp_path, seed, epochs, batch_size):
    p=payload();p['controls'].update(seed=seed,epochs=epochs,batch_size=batch_size)
    p['calibration']['sparse'][0]['calibration_only']=1.
    p['query']['sparse'][0]['query_only']=1.
    populations=[store(tmp_path,n,p[n]) for n in ('training','calibration','query')]
    expected=reference_execute(prepare(p))
    actual=execute(*populations,p['controls'])
    assert actual==expected


def test_models_history_vocabulary_and_read_coverage(tmp_path, monkeypatch):
    p=payload();p['controls'].update(epochs=4,batch_size=3)
    training=store(tmp_path,'training',p['training'])
    expected=reference_fit(p['training']['videos'],p['training']['sparse'],p['training']['labels'],**p['controls'])
    reads=[];original=FeatureShards.__getitem__
    def observed(self,index):
        if self is training and -len(self)<=index<len(self):reads.append(index)
        return original(self,index)
    monkeypatch.setattr(FeatureShards,'__getitem__',observed)
    actual=fit(training,**p['controls'])
    # Vocabulary pass, five complete loss passes and four complete shuffled updates.
    assert len(reads)==len(training)*(1+5+4)
    assert all(reads.count(i)==10 for i in range(len(training)))
    assert actual.history==expected.history and len(actual.history)==5
    assert actual.vectorizer.vocabulary_==expected.vectorizer.vocabulary_
    for key,value in actual.neural.state_dict().items():
        torch.testing.assert_close(value,expected.neural.state_dict()[key],rtol=0,atol=0)
    for a,b in zip(actual.baseline,expected.baseline,strict=True):
        np.testing.assert_array_equal(a.coef_,b.coef_)
    query=store(tmp_path,'query',p['query'])
    before=dict(actual.vectorizer.vocabulary_)
    actual.predict(query)
    assert actual.vectorizer.vocabulary_==before


def test_group_leakage_rejected_before_training(tmp_path, monkeypatch):
    from sciona import video_multilabel_sharded_training as module
    p=payload();p['query']['groups'][0]=p['training']['groups'][0]
    populations=[store(tmp_path,n,p[n]) for n in ('training','calibration','query')]
    monkeypatch.setattr(module,'fit',lambda *a,**k:pytest.fail('Training before group validation'))
    with pytest.raises(ValueError,match='overlap'):execute(*populations,p['controls'])


@pytest.mark.parametrize('controls', [{'epochs':0},{'batch_size':True},{'learning_rate':float('inf')}])
def test_invalid_training_controls(tmp_path,controls):
    p=payload();p['controls'].update(controls)
    training=store(tmp_path,'training',p['training'])
    with pytest.raises(ValueError):fit(training,**p['controls'])


def test_explicit_threshold_functions_are_used(tmp_path):
    from sciona.atoms.ml.calibration.empirical_f1 import fit_empirical_f1_thresholds, apply_column_thresholds
    p=payload();p['controls']['epochs']=2
    populations=[store(tmp_path,n,p[n]) for n in ('training','calibration','query')]
    calls=[]
    def fit_thresholds(scores,labels):
        calls.append(('fit',scores.shape,labels.dtype))
        return fit_empirical_f1_thresholds(scores,labels)
    def apply_thresholds(scores,thresholds):
        calls.append(('apply',scores.shape,thresholds.dtype))
        return apply_column_thresholds(scores,thresholds)
    result=execute(*populations,p['controls'],threshold_fit_function=fit_thresholds,threshold_apply_function=apply_thresholds)
    assert result==reference_execute(prepare(p))
    assert calls==[('fit',(4,2),np.dtype('int64')),('apply',(2,2),np.dtype('float64'))]


def test_existing_mean_provider_is_used_per_label(tmp_path):
    from sciona.atoms.ml.model_selection.materialized_prediction import average
    p=payload();p['controls']['epochs']=2
    populations=[store(tmp_path,n,p[n]) for n in ('training','calibration','query')]
    calls=[]
    def mean(vectors):
        calls.append([v.shape for v in vectors])
        result=average(vectors)
        np.testing.assert_array_equal(result,(vectors[0]+vectors[1])/2)
        return result
    assert execute(*populations,p['controls'],mean_function=mean)==reference_execute(prepare(p))
    assert calls==[[(4,),(4,)],[(4,),(4,)],[(2,),(2,)],[(2,),(2,)]]
