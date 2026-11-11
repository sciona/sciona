"""Compare independent networks against pinned source with identical synthetic weights."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.amex_network import Network
SOURCE_SHA='cc56e0f2816a87ac7cde60f04e07c5de8bc4d18c183dfa79454f2a255d771f96'


def main(path):
    if not __debug__:raise RuntimeError('Assertions required')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA:raise ValueError('Source pin differs')
    nodes=[n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef) and n.name=='Amodel'];assert len(nodes)==1
    ns={'__name__':'pinned_network','torch':torch,'nn':nn,'__builtins__':{'__build_class__':__build_class__,'super':super,'range':range,'enumerate':enumerate,'int':int}}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pinned network source>','exec'),ns)
    aliases={'sequence_projection':'input_series_block','feature_projection':'input_feature_block','recurrence':'gru_series','feature_hidden':'hidden_feature_block','output':'output_block'}
    cases=0
    for sw,fw,hw in ((4,6,8),(223,12776,128)):
        for combined in (False,True):
            torch.manual_seed(113);ours=Network(sw,fw,combined=combined,hidden_width=hw).eval()
            source=ns['Amodel'](sw,fw,1,3,hw,use_series_oof=combined).eval()
            source.load_state_dict({aliases[k.split('.')[0]]+'.'+k.split('.',1)[1]:v for k,v in ours.state_dict().items()})
            series=torch.randn(3,13,sw);features=torch.randn(3,fw);mask=(torch.arange(13)[None,:]<torch.tensor([3,13,1])[:,None]).float()
            a=series.clone().requires_grad_();b=series.clone().requires_grad_()
            actual=ours(a,mask,features);expected=source(dict(batch_series=b,batch_mask=mask,batch_feature=features))
            torch.testing.assert_close(actual,expected,atol=1e-7,rtol=1e-6)
            actual.sum().backward();expected.sum().backward()
            torch.testing.assert_close(a.grad,b.grad,atol=1e-7,rtol=1e-5)
            cases+=1
    files=[ROOT/'sciona/amex_network.py',ROOT/'tests/test_amex_network.py',Path(__file__).resolve()]
    report=dict(status='passed',approved=False,catalog_mutations=0,synthetic_only=True,source_software_sha256=SOURCE_SHA,
        checks=dict(network_cases=cases,both_variants=True,source_input_dimensions=True,last_valid_timestep_pooling=True,input_gradients=True),
        sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        limits=['Identical synthetic weights and eval-mode comparison on current CPU PyTorch; historical GPU training and learned-weight parity unqualified.',
            'Generalized input dimensions supported explicitly; full optimizer, checkpoint and epoch lifecycle pending.'])
    (ROOT/'docs/reviews/competition_amex_network_comparison.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['checks']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-code',type=Path,required=True);main(p.parse_args().source_code)
