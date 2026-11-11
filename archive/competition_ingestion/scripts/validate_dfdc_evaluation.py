"""Source validation-loop and independent clip/class weighting checks."""
import argparse
import ast
from collections import defaultdict
from contextlib import redirect_stdout
import hashlib
import io
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import warnings

import numpy as np
import sklearn
from sklearn.metrics import log_loss
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.dfdc_evaluation import score_videos,evaluate,ValidationDataset
from sciona.dfdc_dataset import make_loader


class CPU(ast.NodeTransformer):
    def visit_Call(self,node):
        self.generic_visit(node)
        if isinstance(node.func,ast.Attribute) and node.func.attr=='cuda':node.func.attr='cpu'
        return node


class Model(torch.nn.Module):
    def forward(self,x):return x


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args()
    pins=json.loads((ROOT/'docs/reviews/competition_dfdc_source_pins.json').read_text())
    p=args.source_root/'training/pipelines/train_classifier.py'
    assert hashlib.sha256(p.read_bytes()).hexdigest()==next(f['sha256'] for f in pins['files'] if f['path']=='training/pipelines/train_classifier.py')
    fn=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='validate')
    fn=CPU().visit(fn);ast.fix_missing_locations(fn)
    namespace={'defaultdict':defaultdict,'torch':torch,'np':np,'log_loss':log_loss,'tqdm':lambda x:x}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<source-video-validation>','exec'),namespace)
    cases=0
    for batch_size in (1,3,7):
        for dtype in (torch.float32,torch.float64):
            logits=torch.tensor([-2.,-.5,.1,.4,-1.,1.,2.,-.8,1.3],dtype=dtype).reshape(-1,1)
            labels=torch.tensor([0,0,0,0,0,1,1,1,1]).reshape(-1,1)
            groups=torch.tensor([0,0,1,0,1,2,2,3,2])
            source_batches=[];adapted_batches=[]
            for start in range(0,len(logits),batch_size):
                stop=start+batch_size
                base={'image':logits[start:stop],'labels':labels[start:stop]}
                source_batches.append(dict(base,img_name=[f'synthetic-{int(i)}/sample' for i in groups[start:stop]]))
                adapted_batches.append(dict(base,clip_position=groups[start:stop]))
            model=Model().eval()
            with redirect_stdout(io.StringIO()):expected,_,_=namespace['validate'](model,source_batches)
            actual=evaluate(model,adapted_batches)
            assert actual['loss']==float(expected) and actual['clips']==4 and actual['samples']==9
            cases+=1
    # Unequal sample counts and unequal class clip counts must not change class weights.
    probs=[.2]*8+[.6]+[.7,.9];labels=[0]*9+[1,1];groups=[0]*8+[1]+[2,2]
    actual=score_videos(probs,labels,groups)
    expected_real=(-math.log(.8)-math.log(.4))/2
    expected_fake=-math.log(.8)
    assert abs(actual['loss']-(expected_real+expected_fake)/2)<1e-14
    assert abs(actual['loss']-log_loss(labels,probs))>1e-3
    rejected=0
    for values,labels,groups in (([.2,.3],[0,0],[0,1]),([.2,.8],[0,1],[0,0]),
                                 ([float('nan'),.8],[0,1],[0,1]),([],[],[])):
        try:score_videos(values,labels,groups)
        except ValueError:rejected+=1
        else:raise AssertionError('invalid validation population accepted')
    # Group identity remains positional through real preparation/validation batching.
    records=[{'image':np.full((19,23,3),70+i,dtype=np.uint8),'label':i%2,'fold':0,
              'frame':i*20,'clip_position':i//2,'mask':None,'landmarks':None} for i in range(6)]
    # Give each clip a consistent label for downstream metrics.
    for row in records:row['label']=row['clip_position']%2
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        dataset=ValidationDataset(records);state=np.random.get_state()
        try:
            dataset.reset(1,111)
            batches=list(make_loader(dataset,batch_size=2))
            observed=torch.cat([b['clip_position'] for b in batches]).tolist()
            assert observed==[records[i]['clip_position'] for i in dataset.indices]
            assert [len(b['image']) for b in batches]==[4,2]
        finally:np.random.set_state(state)
    files=['sciona/dfdc_evaluation.py','sciona/dfdc_dataset.py','scripts/validate_dfdc_evaluation.py',
           'docs/reviews/competition_dfdc_source_pins.json']
    report={'format':'dfdc-evaluation-validation.v1','result':'passed','source_commit':pins['commit'],
       'dependency':{'scikit_learn':sklearn.__version__},
       'checks':{'source_validation_loop_cases':cases,'independent_clip_and_class_weighting':True,
                 'invalid_validation_rejections':rejected,'validation_loader_group_positions':True},
       'semantics':['Average sigmoid predictions within each clip, then class-specific clip logloss, then equal class mean.',
                    'No confidence heuristic in validation; that belongs to final inference.',
                    'evaluate_val uses strict lower loss for improvement, despite best_dice filename; outer checkpoint lifecycle remains to implement.'],
       'adaptations':['Positional clip groups replace media-name parsing; no prediction files or identity outputs.',
                      'Require both classes, consistent per-clip labels and finite probabilities.',
                      'CPU source loop adaptation; both paths use installedsklearn clipping behavior.'],
       'limits':'Synthetic models/populations and actual validationDataset preparation. No full B7 validation lifecycle, historicalsklearn binary or promotion claim.',
       'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}}
    (ROOT/'docs/reviews/competition_dfdc_evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
