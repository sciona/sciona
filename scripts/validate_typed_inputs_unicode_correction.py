"""Qualify exact descriptor round trips after rejecting surrogate code points."""
import inspect
import json
from pathlib import Path
import subprocess

from sciona.atoms.ml.pipeline import typed_inputs as provider
from scripts import validate_typed_inputs_provider as previous

ROOT=previous.ROOT
ORIGINAL_REVISION='a4b3000'
PROVIDER_PATH='src/sciona/atoms/ml/pipeline/typed_inputs.py'


def validate():
    repo=ROOT.parent/'sciona-atoms-ml'
    original_source=subprocess.check_output(['git','show',ORIGINAL_REVISION+':'+PROVIDER_PATH],cwd=repo)
    import ast
    import hashlib
    tree=ast.parse(original_source)
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='encode_feature_descriptors')
    function.decorator_list=[]
    # Execute only the frozen pure encoder function, without registry mutation.
    namespace={'json':json}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'<frozen-synthetic-encoder>','exec'),namespace)
    bad_pair=[['\ud834\udd1e']]
    encoded=namespace['encode_feature_descriptors'](bad_pair)
    if json.loads(encoded[0])==bad_pair[0]:raise ValueError('Frozen counterexample no longer reproduces')
    execution=previous.validate()
    rejected=[]
    for name,value in [('surrogate_pair',bad_pair),('lone_high_surrogate',[['\ud834']]),
                       ('lone_low_surrogate',[['\udd1e']])]:
        try:provider.encode_feature_descriptors(value)
        except ValueError:rejected.append(name)
        else:raise ValueError('Surrogate code point accepted')
    values=[['scalar','\U0001d11e'],['scalar','\U0001f680'],['scalar','\uffff'],
            ['controls','\x00\n\t'],['combining','e\u0301'],['composed','\u00e9']]
    actual=provider.encode_feature_descriptors(values)
    if [json.loads(item) for item in actual]!=values or len(set(actual))!=len(values):
        raise ValueError('Valid Unicode scalar descriptors changed')
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,
        frozen_counterexample=dict(reproduced=True,original_provider_revision=ORIGINAL_REVISION,
            original_provider_sha256=hashlib.sha256(original_source).hexdigest(),
            claim_contradicted='Exact descriptor round trip for arbitrary nonblank Python strings',
            input_codepoints=2,decoded_codepoints=1),
        prior_conversion_checks=execution,rejected_surrogate_contracts=rejected,
        valid_unicode_roundtrip_cases=len(values),
        provider_sha256=previous.sha(inspect.getfile(provider)),validator_sha256=previous.sha(__file__),
        limitations=['Corrected encoder requires Unicode scalar-value strings; explicit surrogate code points are rejected.',
            'Existing draft evidence pins the older provider and must be superseded through reviewed intake before publication.',
            'Original frozen reports remain historical evidence; no catalog approval or whole-CDG approval is claimed.'])


if __name__=='__main__':
    result=validate()
    (ROOT/'docs/reviews/typed_inputs_unicode_correction.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,counterexample_reproduced=True,surrogate_rejections=3,unicode_cases=6,approved=False)))
