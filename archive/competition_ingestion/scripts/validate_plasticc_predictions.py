"""Independent synthetic row oracle and original-source prediction comparison."""
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona import plasticc_predictions as runtime


def validate(root):
    path=root/'avocado/plasticc.py'
    pins=json.loads((ROOT/'docs/reviews/competition_plasticc_source_pins.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==pins['files']['avocado/plasticc.py']
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='create_kaggle_predictions')
    namespace={'np':np}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    counters=dict(source_cases=0,independent_rows=0,input_preservation=0,default_input=0,
                  zero_mass_semantics=0,missing_required_class=0)
    rng=np.random.default_rng(51)
    classes=[6,15,16,42,52,53,62,64,65,67,88,90,92,95]
    for case in range(12):
        columns=classes+([99] if case%2 else [])
        values=rng.uniform(.001,1,(12,len(columns))); values/=values.sum(axis=1)[:,None]
        index=['synthetic-%d'%i for i in range(12)]
        predictions=pd.DataFrame(values,index=index,columns=columns)
        metadata=pd.DataFrame({'galactic':np.resize([True,False],12)},index=index)
        if case%3==0: metadata=metadata.iloc[::-1]  # Label alignment, not positional masking.
        dataset=SimpleNamespace(metadata=metadata,predictions=predictions)
        original=predictions.copy(deep=True)
        actual=runtime.create_kaggle_predictions(dataset,predictions)
        expected=namespace['create_kaggle_predictions'](dataset,predictions)
        pd.testing.assert_frame_equal(actual,expected)
        pd.testing.assert_frame_equal(predictions,original)
        pd.testing.assert_frame_equal(actual,runtime.create_kaggle_predictions(dataset))
        for key,row in original.iterrows():
            values=row.to_dict()
            if 99 in values:
                values[99]=0.; total=sum(values.values()); values={k:v/total for k,v in values.items()}
            galactic=metadata.loc[key,'galactic']
            for label in values:
                if (label in [6,16,53,65,92]) != galactic: values[label]=0.
            values[99]=.04 if galactic else values[42]+.6*values[62]+.2*values[52]+.2*values[95]
            total=sum(values.values())
            np.testing.assert_allclose(actual.loc[key],[values[k]/total for k in actual.columns],rtol=1e-13,atol=1e-13)
            counters['independent_rows']+=1
        np.testing.assert_allclose(actual.sum(axis=1),1)
        assert (actual.to_numpy()>=0).all()
        counters['source_cases']+=1; counters['input_preservation']+=1; counters['default_input']+=1
    # Degenerate source behavior is not silently imputed; graph boundary must validate it.
    zeros=pd.DataFrame(0.,index=['synthetic'],columns=classes)
    ds=SimpleNamespace(metadata=pd.DataFrame({'galactic':[False]},index=zeros.index))
    assert runtime.create_kaggle_predictions(ds,zeros).isna().all().all()
    ds.metadata['galactic']=True
    output=runtime.create_kaggle_predictions(ds,zeros)
    assert output.loc['synthetic',99]==1.
    counters['zero_mass_semantics']+=2
    try: runtime.create_kaggle_predictions(ds,zeros.drop(columns=42))
    except KeyError: counters['missing_required_class']+=1
    else: raise AssertionError('Missing required class accepted')
    paths=[Path(runtime.__file__),Path(__file__),ROOT/'docs/reviews/competition_plasticc_source_pins.json',ROOT/'docs/licenses/Avocado-MIT.txt']
    return dict(status='passed',approved=False,checks=counters,
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        scope='Source task-specific adjustment; synthetic probabilities only',
        limitations=['Not general unknown-object detection or Platt calibration',
                     'Full runtime must validate required classes, alignment and nonzero output mass'])

if __name__=='__main__':
    result=validate(Path(sys.argv[1]) if len(sys.argv)>1 else Path('/private/tmp/sciona_plasticc_source'))
    (ROOT/'docs/reviews/competition_plasticc_predictions.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['checks'],sort_keys=True))
