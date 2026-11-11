"""Synthetic full-state conversion and neural-output parity for all backbones."""
import gc
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.adversarial_state_mapping import mapping,convert_state,SCOPES
from sciona.adversarial_inception_v3 import build_inception_v3
from sciona.adversarial_inception_v4 import build_inception_v4
from sciona.adversarial_inception_resnet import build_inception_resnet
from sciona.adversarial_resnet import build_resnet101


def main():
    torch.set_num_threads(2)
    builders={'inception_v3':build_inception_v3,'inception_v4':build_inception_v4,
              'inception_resnet':build_inception_resnet,'resnet101':build_resnet101}
    counts={};rejects=0
    for family,build in builders.items():
        model=build(seed=919).requires_grad_(False)
        state=model.state_dict()
        scope=SCOPES[family][0]
        names=mapping(model,family,scope)
        # Explicit known source-name anchors, independent of mapping generation.
        if family=='resnet101':
            assert names['root.weight']==('resnet_v2_101/conv1/weights','HWIO')
            assert names['blocks.2.22.bn2.gamma'][0]=='resnet_v2_101/block3/unit_23/bottleneck_v2/conv2/BatchNorm/gamma'
        else:
            assert names['layers.0.conv.weight']==(scope+'/Conv2d_1a_3x3/weights','HWIO')
            assert names['layers.0.norm.beta'][0]==scope+'/Conv2d_1a_3x3/BatchNorm/beta'
            assert not any(k.endswith('.gamma') for k in names)
        source={}
        for key,(name,layout) in names.items():
            v=state[key]
            source[name]=(v.permute(2,3,1,0) if layout=='HWIO' else v.T if layout=='IO' else v).clone()
        converted=convert_state(model,source,family=family,scope=scope)
        for key in state:
            assert torch.equal(converted[key],state[key]) and converted[key].is_contiguous()
            assert converted[key].data_ptr()!=source[names[key][0]].data_ptr()
        for alias in SCOPES[family]:
            aliases=mapping(model,family,alias)
            assert all(v[0].startswith(alias+'/') for v in aliases.values())
            assert [v[0].split('/',1)[1] for v in aliases.values()]==[v[0].split('/',1)[1] for v in names.values()]
        x=torch.linspace(-1,1,3*299*299).reshape(1,3,299,299).requires_grad_()
        y,e=model(x);g,=torch.autograd.grad(y[:,17].sum(),x)
        restored=build(seed=920,state=converted).requires_grad_(False)
        z,f=restored(x);h,=torch.autograd.grad(z[:,17].sum(),x)
        assert torch.equal(y,z) and torch.equal(g,h)
        if 'AuxLogits' in e:assert torch.equal(e['AuxLogits'],f['AuxLogits'])
        first=next(iter(source));bad=dict(source);bad.pop(first)
        for invalid in (bad,{**source,'unexpected':torch.zeros(1)}, {**source,first:source[first].double()}):
            try:convert_state(model,invalid,family=family,scope=scope)
            except ValueError:rejects+=1
            else:raise AssertionError('invalid source state accepted')
        counts[family]={'tensors':len(names),'scopes':len(SCOPES[family]),'full_output_input_gradient_exact':True}
        del model,restored,state,source,converted,bad,y,z,e,f,g,h,x,invalid
        gc.collect()
    paths=['sciona/adversarial_state_mapping.py','scripts/validate_adversarial_state_mapping.py',
           'sciona/adversarial_inception_v3.py','sciona/adversarial_inception_v4.py',
           'sciona/adversarial_inception_resnet.py','sciona/adversarial_resnet.py',
           'sciona/adversarial_inception_v3_topology.json','sciona/adversarial_inception_v4_topology.json',
           'sciona/adversarial_inception_resnet_topology.json','docs/reviews/competition_adversarial_source_pins.json']
    report={'format':'adversarial-state-mapping-validation.v1','result':'passed','checks':counts,
            'invalid_states_rejected':rejects,
            'limits':'Complete synthetic inverse-layout round trips and source-name anchors; no real pretrained checkpoint or TensorFlow checkpoint binary reader. Caller must supply exact decoded model variables; optimizer slots and unrelated tensors rejected. Historical variable names remain source-derived rather than checkpoint-verified.',
            'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}}
    (ROOT/'docs/reviews/competition_adversarial_state_mapping.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'families':counts,'rejected':rejects}))


if __name__=='__main__':main()
