"""Exhaustive short synthetic structure comparison against pinned source."""
import ast
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from sciona.openvaccine_structure import loop_labels

ROOT=Path(__file__).resolve().parents[1]


@lru_cache(None)
def structures(length):
    if length==0:return ('',)
    result=['.'+s for s in structures(length-1)]
    for inside in range(length-1):
        result.extend('('+a+')'+b for a in structures(inside) for b in structures(length-2-inside))
    return tuple(result)


def main():
    cache=Path('/private/tmp/sciona_openvaccine_source')
    manifest=json.loads((cache/'manifest.json').read_text())
    name='scripts/nullrecurrent_inference.py';raw=(cache/name).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==next(p['sha256'] for p in manifest['pins'] if p['software_path']==name)
    names={'convert_structure_to_bps','secstruct_to_partner','write_bprna_string'}
    definitions=[n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in definitions}==names
    ns=dict(np=np,re=re)
    exec(compile(ast.Module(body=definitions,type_ignores=[]),'<source-loop-assignment>','exec'),ns)
    count=0;observed=set()
    for length in range(1,12):
        for structure in structures(length):
            actual=loop_labels(structure)
            expected=ns['write_bprna_string'](structure)
            assert actual==expected,(structure,actual,expected)
            observed.update(actual);count+=1
    assert observed==set('SEHBIM')
    invalid=['','(',')','(..','..)','([..])','abc',None]
    for value in invalid:
        try:loop_labels(value)
        except ValueError:pass
        else:raise AssertionError('Invalid structure accepted')
    report=dict(status='passed',synthetic_only=True,source_commit=manifest['commit'],
                exhaustive_structures=count,max_length=11,observed_labels=sorted(observed),
                invalid_cases=len(invalid),
                scope='Independent loop labeling matches all balanced dot-parenthesis structures through length11; pseudoknots rejected, no folding engine execution.',
                runtime_sha256=hashlib.sha256((ROOT/'sciona/openvaccine_structure.py').read_bytes()).hexdigest(),
                validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'docs/reviews/competition_openvaccine_structure.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
