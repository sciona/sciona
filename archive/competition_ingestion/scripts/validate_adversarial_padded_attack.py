"""Full forty-step padded targeted attack with explicit synthetic model states."""
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.adversarial_ensemble import build_ensemble,ORDER
from sciona.adversarial_state_mapping import mapping
from sciona.adversarial_rgb_attack import attack_rgb
from sciona.adversarial_image_boundary import source_rgb


def main():
    torch.set_num_threads(2)
    # Explicit synthetic supplied states: nonzero frozen BN offsets keep zero-pad
    # input derivatives defined. Default all-zero offsets can yield undefined
    # normalization on zero padding; do not hide that by changing the algorithm.
    models=build_ensemble('targeted_small',{s:{'kind':'random','seed':1400+i} for i,s in enumerate(ORDER['targeted_small'])})
    specs={}
    for model,scope,family in zip(models,ORDER['targeted_small'],('inception_v3','inception_resnet')):
        with torch.no_grad():
            for name,buffer in model.named_buffers():
                if name.endswith('.beta'):buffer.fill_(.1)
        state=model.state_dict();source={}
        for key,(name,layout) in mapping(model,family,scope).items():
            value=state[key]
            source[name]=(value.permute(2,3,1,0) if layout=='HWIO' else value.T if layout=='IO' else value).clone()
        specs[scope]={'kind':'slim_tensors','seed':1401,'tensors':source}
    del models,model,state,source,value,buffer
    rgb=(np.arange(299*299*3,dtype=np.uint32)%256).astype(np.uint8).reshape(1,299,299,3)
    original=rgb.copy();rng=torch.get_rng_state().clone()
    result=attack_rgb(rgb,mode='targeted',epsilon=4,initializations=specs,targets=np.array([17],np.int64))
    assert result['images'].shape==(1,299,299,3) and len(result['pngs'])==1
    assert len(result['batch_losses'])==1 and len(result['batch_losses'][0])==40
    assert result['labels'].tolist()==[17]
    normalized=(rgb.astype(np.float64)/255.*2.-1.).astype(np.float32)
    assert np.max(np.abs(result['images']-normalized))<=2*4/255+1e-7
    assert np.any(result['images']!=normalized)
    np.testing.assert_array_equal(np.array(Image.open(BytesIO(result['pngs'][0]))),source_rgb(result['images'][0]))
    np.testing.assert_array_equal(rgb,original)
    assert torch.equal(rng,torch.get_rng_state())
    paths=['sciona/adversarial_rgb_attack.py','scripts/validate_adversarial_padded_attack.py',
           'sciona/adversarial_image_boundary.py','sciona/adversarial_lifecycle.py','sciona/adversarial_ensemble.py',
           'sciona/adversarial_state_mapping.py','docs/reviews/competition_adversarial_lifecycle.json',
           'docs/reviews/competition_adversarial_image_boundary.json']
    report={'format':'adversarial-padded-attack-validation.v1','result':'passed',
            'checks':{'full_neural_batch_size':10,'real_entries':1,'padded_entries':9,
                      'targeted_small_full_iterations':40,'full_models':2,'caller_slim_states_used':True,
                      'only_real_entry_encoded':True,'decoded_PNG_matches_source_scaling':True,
                      'normalized_projection_bound':True,'input_and_rng_unchanged':True},
            'limits':'Synthetic supplied model states with BN beta=.1, no pretrained claim. Full padded targeted-small lifecycle; padded8/5-model branches not tested here. Source saved PNG stretching does not guarantee pixel epsilon bound. Degenerate zero-normalization states explicitly reject.',
            'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}}
    (ROOT/'docs/reviews/competition_adversarial_padded_attack.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
