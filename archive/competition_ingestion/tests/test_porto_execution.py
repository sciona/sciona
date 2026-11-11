"""Synthetic boundary rejection and detached canonical input checks."""
import hashlib
import json
import pytest
from scripts.porto_synthetic import payload
from sciona.porto_execution import Prepared,prepare,execute


def test_detached_canonical_boundary():
    p=payload(); prepared=prepare(p)
    p['training']['values'][0][0]=999
    assert json.loads(prepared.configuration)['training']['values'][0][0]!=999
    assert prepare(json.loads(prepared.configuration))==prepared
    assert 'configuration' not in repr(prepared)


@pytest.mark.parametrize('case',['overlap','duplicate','labels','extra','models','roles','binary','seed','nan'])
def test_invalid_boundary(case):
    p=payload()
    if case=='overlap':p['query']['identities'][0]=p['training']['identities'][0]
    if case=='duplicate':p['query']['identities'][1]=p['query']['identities'][0]
    if case=='labels':p['training']['labels'][0]=True
    if case=='extra':p['query']['labels']=[0]*8
    if case=='models':p['controls']['neural_models'][1]=p['controls']['neural_models'][0]
    if case=='roles':p['controls']['preparation']['dropped_columns']=[3]
    if case=='binary':p['controls']['preparation']['binary_columns']=[0]
    if case=='seed':p['controls']['tree']['seed']=True
    if case=='nan':p['query']['values'][0][0]=float('nan')
    with pytest.raises(ValueError):prepare(p)


def test_integrity_before_execution():
    p=prepare(payload())
    with pytest.raises(ValueError):execute(Prepared(p.configuration+' ',p.fingerprint))
    changed=p.configuration+' '
    with pytest.raises(ValueError):execute(Prepared(changed,hashlib.sha256(changed.encode()).hexdigest()))
