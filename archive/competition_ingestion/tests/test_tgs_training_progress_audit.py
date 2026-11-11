"""Synthetic ledger rejection tests; no model-training claims."""
import copy
import hashlib
import json

import pytest
from scripts.audit_tgs_training_progress import audit_fit


def example():
    fit=dict(key='synthetic',branch='keras',epoch_ceiling=5,
             controls=dict(batch_size=2,callback='reduce_lr',learning_rate=.01,
                           early_stop_patience=2,reduce_lr_patience=10,reduce_lr_factor=.5,reduce_lr_min=.001))
    result=dict(fit='synthetic',fit_contract_sha256=hashlib.sha256(json.dumps(fit,sort_keys=True).encode()).hexdigest(),
                actual_epochs=3,optimizer_updates=12,steps_per_epoch=4,
                population_counts=dict(training_labeled=4,training_query=0,validation_labeled=1),
                history=[dict(epoch=i,learning_rate=.01,source_metric=0.) for i in range(3)],
                best_checkpoint={'synthetic':True},periodic_checkpoints={})
    return fit,result


def test_exact_early_stop_history_is_accepted():
    fit,result=example()
    assert audit_fit(fit,result)==[result['best_checkpoint']]


def test_coherently_shortened_counts_still_need_source_stop():
    fit,result=example()
    result.update(actual_epochs=2,optimizer_updates=8,history=result['history'][:2])
    with pytest.raises(ValueError,match='without source early stopping'):
        audit_fit(fit,result)


def test_counter_and_learning_rate_tampering_fail():
    fit,result=example()
    invalid=copy.deepcopy(result);invalid['optimizer_updates']=11
    with pytest.raises(ValueError,match='update count'):
        audit_fit(fit,invalid)
    invalid=copy.deepcopy(result);invalid['history'][1]['learning_rate']=.005
    with pytest.raises(ValueError,match='learning rate'):
        audit_fit(fit,invalid)
