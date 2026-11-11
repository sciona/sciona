"""Full synthetic lifecycle execution over source-validated numerical stages."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import signal
from unittest.mock import patch
import numpy as np
import torch
from sciona.hubmap_encoder import build_encoder
from sciona.hubmap_lifecycle import FoldStage,run_lifecycle
from sciona.hubmap_raw_inference import predict_slide


def validate(root):
    torch.set_num_threads(1);torch.manual_seed(367)
    encoder=build_encoder();state=encoder.state_dict();del encoder
    slides=[]
    for i in range(3):
        image=np.random.RandomState(373+i).randint(0,256,(35,37,3)).astype(np.uint8)
        mask=np.zeros((35,37),dtype=np.uint8)
        if i!=1:mask[::2]=1
        slides.append((image,mask))
    def stage(seed):
        options=dict(start_epoch=18,end_epoch=18,maximum_bin=4,training_batch_size=2,
                     validation_batch_size=2,input_side=32,python_rng=random.Random(seed),
                     numpy_rng=np.random.RandomState(seed+1),torch_generator=torch.Generator().manual_seed(seed+2))
        return FoldStage(state,seed,['synthetic-valid'],options)
    initial=stage(0);retraining=stage(1)
    pseudo=np.random.RandomState(379).randint(0,256,(35,37,3)).astype(np.uint8)
    final=np.random.RandomState(383).randint(0,256,(35,37,3)).astype(np.uint8)
    calls=[]
    def observed(models,image,**options):
        result=predict_slide(models,image,**options)
        calls.append(dict(tta=options['tta'],models=len(models),foreground_pixels=int(result.sum())))
        return result
    with patch('sciona.hubmap_lifecycle.predict_slide',side_effect=observed):
        result=run_lifecycle(slides,['synthetic-train-a','synthetic-train-b','synthetic-valid'],[[pseudo]],[final],
            initial_stages=[initial],retraining_stages=[retraining],initial_selectors=[(0,'best_loss')],
            final_selectors=[(0,'best_loss')],tile_size=32,
            pseudo_rng=dict(python_rng=random.Random(389),torch_generator=torch.Generator().manual_seed(397)),
            final_rng=dict(python_rng=random.Random(401),torch_generator=torch.Generator().manual_seed(409)),
            inference_geometry=dict(resolution=32,input_resolution=32,pad_size=8,batch_size=4))
    assert [call['tta'] for call in calls]==[4,3]
    assert len(result['checkpoints'])==len(result['predictions'])==1
    assert result['predictions'][0].shape==final.shape[:2]
    assert result['predictions'][0].dtype==np.uint8
    assert np.isin(result['predictions'][0],[0,1]).all()
    assert len(result['initial_epochs'])==len(result['retraining_epochs'])==1
    paths=sorted(root.glob('sciona/hubmap_*.py'))+[root/'scripts/validate_hubmap_lifecycle.py']
    return dict(approved=False,synthetic_only=True,full_lifecycle_executed=True,inference_calls=calls,
                initial_epochs=result['initial_epochs'][0],retraining_epochs=result['retraining_epochs'][0],
                retained_checkpoints=len(result['checkpoints']),
                implementation_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                limitations=['Full real numerical stages executed; this is an integration smoke check, not an independent entire-lifecycle source comparison.',
                             'Synthetic encoder state and one fold/epoch per stage at small tile/input sizes; source-sized stages have separate evidence.',
                             'Ordered multi-fold checkpoint routing and full serialized graph/provider/lineage approval gates remain.'])


if __name__=='__main__':
    signal.alarm(120)
    root=Path(__file__).resolve().parents[1]
    report=validate(root)
    (root/'docs/reviews/competition_hubmap_lifecycle.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'implementation_sha256','limitations'}}))
