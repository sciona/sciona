"""Source arithmetic checks for synthetic checkpoint and three-model predictions."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sciona.webtraffic_ensemble import average_checkpoint_predictions, finalize_predictions


def validate(root,source):
    pin=json.loads((root/'docs/reviews/competition_webtraffic_notebook_pin.json').read_text())
    source_pins=json.loads((root/'docs/reviews/competition_webtraffic_source_pins.json').read_text())
    assert hashlib.sha256((source/'trainer.py').read_bytes()).hexdigest()==source_pins['files']['trainer.py']
    code=(source/'submission-final-code.py').read_text()
    assert hashlib.sha256(code.encode()).hexdigest()==pin['code_sha256']
    # Parse code cells without executing notebook IO, plotting or model construction.
    tree=ast.parse('\n'.join(line for line in code.splitlines() if not line.startswith('%')))
    wanted={'preds','missing_pages','rmdf','f_preds'}
    nodes=[]
    for node in tree.body:
        if not isinstance(node,ast.Assign):continue
        target=node.targets[0]
        name=target.id if isinstance(target,ast.Name) else None
        if isinstance(target,ast.Subscript) and isinstance(target.value,ast.Name):name=target.value.id
        if name in wanted:nodes.append(node)
    assert len(nodes)==6
    class LegacyFrame(pd.DataFrame):
        def append(self,other):return pd.concat([self,other])
    counts=dict(ensembles=0,checkpoint_means=0,boundary_values=0)
    for dtype in [np.float32,np.float64]:
        for seed in range(8):
            rng=np.random.RandomState(500+seed)
            pages=pd.Index([f'SyntheticEnsemble{i}' for i in range(7)])
            columns=pd.RangeIndex(6)
            models=[]
            for model in range(3):
                logs=[pd.DataFrame(rng.uniform(-.3,5,(6,6)).astype(dtype),index=pages[:6],columns=columns)
                      for _ in range(10)]
                saved=[f.copy(deep=True) for f in logs]
                actual=average_checkpoint_predictions(logs)
                # Original trainer accumulation order after batch concatenation.
                original=None
                for frame in logs:
                    cp_predictions=pd.DataFrame(np.expm1(frame.to_numpy()),index=frame.index,columns=frame.columns)
                    if original is None:original=cp_predictions
                    else:original+=cp_predictions
                original/=len(logs)
                pd.testing.assert_frame_equal(actual,original)
                oracle=np.mean(np.stack([np.expm1(f.to_numpy()) for f in logs]),axis=0)
                np.testing.assert_allclose(actual,oracle,rtol=2e-7)
                for a,b in zip(logs,saved):pd.testing.assert_frame_equal(a,b)
                models.append(actual);counts['checkpoint_means']+=1
            ns=dict(np=np,pd=pd,t_preds=models,prev=pd.DataFrame(index=pages))
            # Source pandas append is removed in current pandas; adapt only that API.
            first=nodes[0]
            exec(compile(ast.Module(body=[first],type_ignores=[]),'<notebook-mean>','exec'),ns)
            ns['preds']=LegacyFrame(ns['preds'])
            exec(compile(ast.Module(body=nodes[1:],type_ignores=[]),'<notebook-finalize>','exec'),ns)
            actual=finalize_predictions(models,pages)
            pd.testing.assert_frame_equal(actual,ns['f_preds'])
            assert (actual.loc[pages[-1]]==0).all()
            counts['ensembles']+=1
    values=np.array([-.5,0,.499,.5,1.5,2.5,3.5])
    frame=pd.DataFrame([values],index=['SyntheticBoundary'])
    actual=finalize_predictions([frame.copy() for _ in range(3)],frame.index)
    np.testing.assert_array_equal(actual.iloc[0],[0,0,0,0,2,2,4])
    counts['boundary_values']=len(values)
    logframes=[pd.DataFrame([[0.]]),pd.DataFrame([[np.log(9.)]])]
    np.testing.assert_allclose(average_checkpoint_predictions(logframes).iat[0,0],4.,rtol=1e-14)
    assert not np.isclose(np.expm1(np.mean([0,np.log(9.)])),4.)
    paths=['sciona/webtraffic_ensemble.py','scripts/validate_webtraffic_ensemble.py',
           'docs/reviews/competition_webtraffic_notebook_pin.json',
           'docs/reviews/competition_webtraffic_source_pins.json','docs/licenses/WebTraffic-MIT.txt']
    return dict(approved=False,synthetic_only=True,checks=counts,
                implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
                limitations=['Arithmetic only; no TensorFlow checkpoint restoration or neural inference validated.',
                             'Runtime prediction frames require matching unique page/column axes.',
                             'Notebook append adapted to pandas concat; rounding retains source ties-to-even behavior.',
                             'File export and source-specific submission selection not executed.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root)
    (root/'docs/reviews/competition_webtraffic_ensemble.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='implementation_sha256'}))
