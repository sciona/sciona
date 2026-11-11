import copy
import numpy as np
import pytest

from scripts.audit_tgs_handoff import audit_pseudo_round
from sciona.tgs_pseudo_selection import select_pseudo_labels


def example():
    masks=np.zeros((4,101,101),dtype=bool);masks[1:,:,20:50]=True
    confidence=np.array([1.,.98,.8,.99]);nonconstant=np.ones(4,dtype=bool)
    area=masks.sum((1,2),dtype=np.int64)
    return dict(stage=1,masks=masks,confidence=confidence,nonconstant=nonconstant,area=area,
                **select_pseudo_labels(confidence,area,nonconstant))


def test_valid_synthetic_round_ledger():
    result=audit_pseudo_round(1,example())
    assert result['query_count']==4 and result['keras_selected']==3


@pytest.mark.parametrize('field',['area','keras_indices','torch_folds'])
def test_tampered_round_ledger_rejected(field):
    value=copy.deepcopy(example())
    if field=='area':value[field][0]=1
    elif field=='keras_indices':value[field]=np.array([2])
    else:value[field][0]=np.array([2])
    with pytest.raises(ValueError):audit_pseudo_round(1,value)
