"""Compare reusable prediction to original predictor using synthetic I/O."""
import argparse
import ast
import bz2
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sciona.champs_source_runtime import ChampsSourceRuntime
from sciona.champs_prediction import predict_batches


class SyntheticModel(torch.nn.Module):
    def forward(self,atoms,positions,bonds,*rest):
        x=torch.arange(68,dtype=torch.float32)[None,:,None]*.123456789
        return x.expand(atoms.shape[0],68,bonds.shape[1]),None


class Capture(io.BytesIO):
    def __exit__(self,*args):
        self.saved=self.getvalue()
        return False


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    runtime=ChampsSourceRuntime(args.source)
    source=runtime.sources['src/predictor.py'].decode()
    assert source.count('dev = "cuda"')==1
    source=source.replace('dev = "cuda"','dev = "cpu"')
    nodes=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='single_model_predict']
    assert len(nodes)==1
    capture=Capture()
    ns={'torch':torch,'bz2':bz2,'tqdm':lambda x:x,'open':lambda *_:capture,
        'os':SimpleNamespace(path=SimpleNamespace(join=lambda *_:'synthetic-output')),
        'root':'','settings':{'SUBMISSION_DIR':''}}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<source-predictor>','exec'),ns)
    atoms=torch.ones(3,3,3,dtype=torch.long)
    bonds=torch.zeros(3,4,5,dtype=torch.long)
    bonds[:,:3,0]=torch.tensor([1,8,9])
    bonds[:,:3,1]=torch.tensor([68,7,3])
    targets=torch.zeros(3,4,4)
    targets[:,:2,0]=torch.tensor([[5,0],[3,1],[4,2]])
    targets[:,:2,1]=.3333333
    targets[:,:2,2]=1.234567
    targets[:,:2,3]=1.
    packed=(torch.arange(3),atoms,torch.zeros(3,3,5),bonds,torch.ones(3,4),
            torch.zeros(3,1,7,dtype=torch.long),torch.zeros(3,1),torch.zeros(3,1,10,dtype=torch.long),torch.zeros(3,1),targets)
    batches=[tuple(t[:2] for t in packed),tuple(t[2:] for t in packed)]
    ns['single_model_predict'](batches,SyntheticModel(),'synthetic')
    rows=bz2.decompress(capture.saved).decode().splitlines()[1:]
    reference={int(row.split(',')[0]):float(row.split(',')[1]) for row in rows}
    mapping={i:f'synthetic-{i}' for i in range(6)}
    model=SyntheticModel()
    actual=predict_batches(model,batches,mapping)
    assert model.training
    assert actual=={mapping[i]:reference[i] for i in mapping}
    assert list(actual)==list(mapping.values())
    for bad_batches,bad_mapping in [(batches+ [batches[0]],mapping),(batches,{**mapping,6:'missing'})]:
        try: predict_batches(model,bad_batches,bad_mapping)
        except ValueError: pass
        else: raise AssertionError('Invalid row alignment must fail')
        assert model.training
    repo=Path(__file__).resolve().parents[1]
    args.output.write_text(json.dumps({'source_commit':runtime.commit,'synthetic_predictions':6,
        'source_predictor_exact_match':True,'partial_final_batch':True,'reordered_ids_aligned':True,
        'six_decimal_source_boundary':True,'duplicate_missing_rows_rejected':True,
        'scope':'Original predictor function with CPU device and synthetic I/O, compared to reusable unscaling/alignment adapter; no trained full-ensemble inference claim.',
        'runtime_sha256':hashlib.sha256((repo/'sciona/champs_prediction.py').read_bytes()).hexdigest(),
        'validator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    print('Source prediction parity passes, including six-decimal precision and partial batches')


if __name__=='__main__':
    main()
