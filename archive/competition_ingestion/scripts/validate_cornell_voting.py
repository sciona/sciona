"""Compare independent array voting with reviewed notebook event processing."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.cornell_voting import vote


def main():
    raw=Path('/private/tmp/sciona_cornell_source/inference.code.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='dcf21b634705e83e3fbfcde448146619688449c53ad5a3e4c8bd4bdce9a34bd9'
    definitions=[]
    for s in json.loads(raw):
        try:tree=ast.parse(s)
        except SyntaxError:continue
        definitions.extend(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('prediction_for_clip','get_post_post_process_predictions'))
    assert len(definitions)==2
    ns=dict(pd=pd,np=np,torch=torch,PERIOD=30,SR=10,TTA=1,device='cpu',
        INV_BIRD_CODE={i:f'synthetic{i}' for i in range(264)},progress_bar=lambda x:x)
    exec(compile(ast.Module(body=definitions,type_ignores=[]),'<reviewed-notebook-events>','exec'),ns)
    clips=np.ones((13,2,264),dtype=np.float32)
    frames=np.zeros((13,2,3001,264),dtype=np.float32)
    rng=np.random.default_rng(3)
    for model in range(13):
        for chunk in range(2):
            for section in range(5):
                for label in range(12):
                    if rng.random()<.45:
                        start=section*500+100
                        frames[model,chunk,start:start+3,label]=.8
    expected=np.zeros((12,264),dtype=np.int16)
    class Model:
        def __init__(self,index):self.index=index;self.chunk=0
        def eval(self):return self
        def __call__(self,inputs):
            index=self.chunk;self.chunk+=1
            return dict(framewise_output=torch.from_numpy(frames[self.index,index])[None,None],
                        clipwise_output=torch.from_numpy(clips[self.index,index])[None,None])
    metadata=pd.DataFrame(dict(site=['synthetic'],audio_id=['synthetic']))
    for model in range(13):
        events=ns['prediction_for_clip'](metadata,np.zeros(590,dtype=np.float32),Model(model),.3,.3)
        windows=ns['get_post_post_process_predictions'](events)
        for row in windows.itertuples():
            window=int(row.row_id.rsplit('_',1)[1])//5-1
            for name in row.birds.split():expected[window,int(name.removeprefix('synthetic'))]+=1
    result=vote(clips,frames,duration_seconds=59)
    np.testing.assert_array_equal(result['window_votes'],expected)
    np.testing.assert_array_equal(result['window_decisions'],expected>=4)
    np.testing.assert_array_equal(result['whole_record_decisions'],(expected>=4).sum(axis=0)>=4)
    report=dict(status='passed',models_compared=13,chunks_per_model=2,classes=264,
        window_counts_exact=True,window_decisions_exact=True,whole_record_decisions_exact=True,
        runtime_sha256=hashlib.sha256((ROOT/'sciona/cornell_voting.py').read_bytes()).hexdigest(),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Synthetic interior events compare independent implementation against notebook event processing. Explicit half-open padding/singleton corrections covered separately; no full model or training claim.')
    (ROOT/'docs/reviews/competition_cornell_voting_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
