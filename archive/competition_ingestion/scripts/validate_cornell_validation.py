"""Compare per-record validation metric to pinned original implementation."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from sklearn.metrics import f1_score
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.cornell_validation import clip_f1


def main():
    root=Path('/private/tmp/sciona_cornell_source');path='src/metrics/sed_f1score_clip.py'
    pins=json.loads((root/'manifest.json').read_text())['pins'];raw=(root/path).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==next(p['sha256'] for p in pins if p['software_path']==path)
    class Metric:
        def __init__(self,**kwargs):self.reset()
        def reset(self):pass
    ns=dict(Metric=Metric,torch=torch,f1_score=f1_score,reinit__is_reduced=lambda f:f,
            sync_all_reduce=lambda *args:lambda f:f)
    nodes=[n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef)]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<source-metric>','exec'),ns)
    rng=np.random.default_rng(5);cases=[]
    for mode in ('empty_correct','empty_false_positive','threshold_boundary','unequal_labels','random','attention_rounding'):
        pred=np.zeros((4,2,264));target=np.zeros_like(pred)
        if mode=='empty_false_positive':pred[:,1,0]=.5
        elif mode=='threshold_boundary':pred[:,1,0]=.5;target[:,:,0]=1
        elif mode=='unequal_labels':
            target[0,:,:7]=1;target[1,:,:2]=1;pred[0,1,:3]=.8;pred[1,0,0]=.8
        elif mode=='attention_rounding':pred[:,1,0]=1+2*np.finfo(np.float32).eps;target[:,:,0]=1
        elif mode=='random':
            pred=rng.uniform(0,1,pred.shape);target[:,0]=rng.integers(0,2,(4,264));target[:,1]=target[:,0]
        metric=ns['SedF1ScoreClip']()
        metric.update((dict(clipwise_output=torch.tensor(pred)),torch.tensor(target)))
        actual=clip_f1(pred,target);expected=metric.compute()
        assert abs(actual-expected)<1e-15
        cases.append(dict(case=mode,source_match=True))
    invalid=0
    for pred,truth in [(np.zeros((0,2,264)),np.zeros((0,2,264))),
                       (np.full((1,2,264),np.nan),np.zeros((1,2,264))),
                       (np.zeros((1,2,264)),np.full((1,2,264),.2))]:
        try:clip_f1(pred,truth)
        except ValueError:invalid+=1
        else:raise AssertionError('Invalid validation data accepted')
    report=dict(status='passed',cases=cases,invalid_inputs_rejected=invalid,
        runtime_sha256=hashlib.sha256((ROOT/'sciona/cornell_validation.py').read_bytes()).hexdigest(),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Exact per-record metric semantics including empty targets and two-clip max pooling. Checkpoint ranking/full model validation remains separate.')
    (ROOT/'docs/reviews/competition_cornell_validation_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='passed',cases=len(cases),invalid_inputs_rejected=invalid)))


if __name__=='__main__':main()
