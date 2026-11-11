"""Private boundary synthetic identity, label and integrity checks."""
import json
import pytest
from scripts.vsb_synthetic import payload
from sciona.vsb_execution import Prepared,prepare,execute


def small():
    p=payload();p['training']['signals']=p['training']['signals'][:3];p['query']['signals']=p['query']['signals'][:3]
    return p


def test_detached_canonical_prepared():
    p=small();v=prepare(p);p['training']['signals'][0][0]=999
    assert json.loads(v.configuration)['training']['signals'][0][0]!=999
    assert prepare(json.loads(v.configuration))==v
    assert 'configuration' not in repr(v)
    with pytest.raises(ValueError):execute(Prepared(v.configuration+' ',v.fingerprint))


@pytest.mark.parametrize('case',['overlap','query_labels','width','labels','nan','length','version','bool_label'])
def test_invalid_boundaries(case):
    p=small()
    if case=='overlap':p['query']['identities'][0]=p['training']['identities'][0]
    if case=='query_labels':p['query']['labels']=[]
    if case=='width':p['query']['signals']=[r[:-1] for r in p['query']['signals']]
    if case=='labels':p['training']['labels'][0][0]=2
    if case=='nan':p['training']['signals'][0][0]=float('nan')
    if case=='length':p['query']['signals'].append(p['query']['signals'][0])
    if case=='bool_label':p['training']['labels'][0][0]=True
    if case=='version':p['version']=True
    with pytest.raises(ValueError):prepare(p)
