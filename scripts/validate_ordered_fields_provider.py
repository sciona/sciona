"""Synthetic cross-disciplinary and identity-preservation checks for field routing."""
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
from sciona.atoms.ml.pipeline import ordered_fields as provider

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate():
    cases=[]
    for application in ('synthetic_motion','synthetic_electrical_power'):
        names=('position','velocity') if application=='synthetic_motion' else ('current','resistance')
        for dtype in (np.float32,np.float64):
            first=np.arange(6,dtype=dtype).reshape(3,2)/8
            second=np.full((3,2),2,dtype=dtype)
            original_first,original_second=first.copy(),second.copy()
            # Mapping insertion order is deliberately opposite to model input order.
            mapping={names[1]:second,'unused':object(),names[0]:first}
            before=list(mapping.items())
            for fields in (list(names),tuple(names)):
                result=provider.select_values(mapping,fields)
                assert type(result) is list and result[0] is first and result[1] is second
                assert list(mapping)==[k for k,_ in before]
                assert all(mapping[k] is value for k,value in before)
                actual=result[0]+.5*result[1] if application=='synthetic_motion' else result[0]**2*result[1]
                expected=first+.5*second if application=='synthetic_motion' else first**2*second
                np.testing.assert_array_equal(actual,expected)
                reverse=provider.select_values(mapping,fields[::-1])
                assert reverse[0] is second and reverse[1] is first
                np.testing.assert_array_equal(first,original_first)
                np.testing.assert_array_equal(second,original_second)
                cases.append(dict(application=application,dtype=dtype.__name__,field_sequence=type(fields).__name__,
                    analytic_result_exact=True,explicit_order_preserved=True,borrowed_values_unchanged=True))
    shared=[]
    result=provider.select_values({'left':shared,'right':shared,'none':None},['right','left','none'])
    assert result[0] is result[1] is shared and result[2] is None
    result.append('outer-only')
    assert shared==[]
    result[0].append('shared-value')
    assert shared==['shared-value']
    # Exact keys are preserved; whitespace is not silently normalized.
    marker=object()
    assert provider.select_values({' x ':marker},[' x '])[0] is marker
    class DictSubclass(dict):pass
    class ListSubclass(list):pass
    rejected=[]
    for label,mapping,fields in [
        ('not_mapping',[],['x']),('mapping_subclass',DictSubclass(x=1),['x']),
        ('nonstring_key',{1:1},['x']),('blank_key',{' ':1,'x':2},['x']),
        ('unordered_fields',{'x':1},{'x'}),('string_fields',{'x':1},'x'),
        ('sequence_subclass',{'x':1},ListSubclass(['x'])),('empty_fields',{'x':1},[]),
        ('nonstring_field',{'x':1},[1]),('blank_field',{'x':1},[' ']),
        ('duplicate_field',{'x':1},['x','x']),('missing_field',{'x':1},['y']),
        ('no_implicit_trim',{' x ':1},['x']),('empty_mapping',{},['x'])]:
        try:provider.select_values(mapping,fields)
        except ValueError:rejected.append(label)
        else:raise AssertionError('Invalid field selection accepted: '+label)
    try:provider.witness_select_values()
    except ValueError:rejected.append('abstract_execution')
    else:raise AssertionError('Abstract field selection accepted')
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,
        cases=cases,rejected_contracts=rejected,borrowed_aliasing_preserved=True,
        outer_list_independent=True,exact_names_preserved=True,
        application_contracts=dict(synthetic_motion='Position in metres plus velocity in metres per second times 0.5 seconds.',
            synthetic_electrical_power='Current in amperes squared times resistance in ohms gives watts. Units are caller declarations.'),
        provider_sha256=sha(inspect.getfile(provider)),validator_sha256=sha(__file__),
        limitations=['Routing only: no value validation, dtype conversion, copying, row alignment or unit inference.',
            'Returned values are borrowed references; downstream mutation affects caller-owned values.',
            'Constructed synthetic applications establish explicit ordering reuse, not empirical effectiveness or original-CDG approval.'])


if __name__=='__main__':
    result=validate()
    (ROOT/'docs/reviews/ordered_fields_provider.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,cases=len(result['cases']),rejected=len(result['rejected_contracts']))))
