"""Compare isolated pinned submission arithmetic using synthetic model outputs."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sciona.prediction_postprocessing import clip_truncate_prediction

ROOT=Path(__file__).resolve().parents[1]
SOURCE='1st Place/Phase 1/submission/utilities.py'
PIN='8b9776ae0e7fdbb48e2bee64ee442ca1d2d4023309f953092dba108f93e6896c'


def validate(source):
    data=(source/SOURCE).read_bytes()
    if hashlib.sha256(data).hexdigest()!=PIN:raise ValueError('Pinned submission software changed')
    tree=ast.parse(data)
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='generate_predictions')
    isolated=ast.Module(body=[function],type_ignores=[])
    namespace={'pd':pd}
    exec(compile(isolated,'pinned_prediction_arithmetic','exec'),namespace)
    # Derive software field labels from the pinned AST, never from real data.
    offset_keys=[n.right.slice.value for n in ast.walk(function.body[0]) if isinstance(n,ast.BinOp)
        and isinstance(n.op,ast.Add) and isinstance(n.right,ast.Subscript)
        and isinstance(n.right.slice,ast.Constant)]
    if len(offset_keys)!=1:raise ValueError('Offset expression differs')
    columns=ast.literal_eval(function.body[-2].value.slice)
    override=next(n for n in function.body if isinstance(n,ast.If))
    populations=ast.literal_eval(override.test.comparators[0])
    if len(populations)!=3:raise ValueError('Override routing differs')
    rng=np.random.default_rng(404)
    n=264
    predictions=rng.uniform(-400,700,size=(4,n))
    predictions[:,-8:]=np.array([-1.9,.9,1.9,3.9,4.9,259.9,299.9,800.])
    offsets=rng.uniform(-100,100,size=n);offsets[-8:]=[0,1,-1,.5,-.5,0,0,0]
    class Predictor:
        feature_names_=['synthetic_prediction']
        def __init__(self,values):self.values=values
        def predict(self,features):
            if len(features)!=n:raise ValueError('Row alignment changed')
            return self.values.copy()
    models={i:Predictor(predictions[i]) for i in range(3)}
    models['global_model']=Predictor(predictions[3])
    local=[clip_truncate_prediction(predictions[i],offsets if i in (0,2) else np.zeros(n),1,299) for i in range(3)]
    blended=clip_truncate_prediction((local[0]+local[1]+local[2]+predictions[3])/4,np.zeros(n),4,260)
    direct=clip_truncate_prediction(local[1],np.zeros(n),4,260)
    routes=['SYNTHETIC']+populations+[p.lower() for p in populations]
    for population in routes:
        frame=pd.DataFrame({name:np.arange(n) for name in columns[:-1]})
        frame[offset_keys[0]]=offsets
        frame['synthetic_prediction']=np.zeros(n)
        actual=namespace['generate_predictions'](models,frame,population)
        expected=direct if population.upper() in populations else blended
        np.testing.assert_array_equal(actual[columns[-1]].to_numpy(),expected)
    # Verify that collapsing clipping/truncation into one final step changes results.
    shortcut=clip_truncate_prediction((predictions[0]+offsets+predictions[1]+predictions[2]+offsets+predictions[3])/4,np.zeros(n),4,260)
    differing=int(np.count_nonzero(shortcut!=blended))
    if not differing:raise ValueError('Fixture failed to distinguish operation ordering')
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,
        cases=len(routes),synthetic_rows_per_case=n,exact_compared_predictions=n*len(routes),
        shortcut_counterexamples=differing,source_sha256=PIN,
        implementation_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['sciona/prediction_postprocessing.py','tests/test_prediction_postprocessing.py','scripts/validate_nasa_first_prediction_arithmetic.py']},
        semantics=['Offset two residual branches before per-branch clipping and integer truncation.',
            'Sum three truncated local branches in source order, add unrounded global predictions and divide by four before final clipping/truncation.',
            'Three case-insensitive population routes override the ensemble with the direct branch.'],
        limits=['Constructed predictor outputs isolate submission arithmetic; no native CatBoost training or feature-extraction parity.',
            'Reusable clipping/truncation supports aligned finite real arrays and explicit bounds; source-specific constants and routing remain in the adapter.',
            'Training-helper rounding differs from submission behavior; full wiring and model-state closure remain outstanding.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    report=validate(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/competition_nasa_first_prediction_arithmetic.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['passed','cases','exact_compared_predictions','shortcut_counterexamples','approved']}))
