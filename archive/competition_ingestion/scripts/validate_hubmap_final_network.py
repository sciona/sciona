"""Full source network parity with batchwise classification shortcut enabled."""
import argparse
import ast
import numpy as np
import hashlib
import json
from pathlib import Path
import signal
import torch
from sciona.hubmap_inference_network import InferenceNetwork
from scripts.validate_hubmap_network import source_model,exact_state


def notebook_nodes(root,notebook):
    pins=json.loads((root/'docs/reviews/competition_hubmap_final_notebook_pins.json').read_text())
    assert hashlib.sha256(notebook.read_bytes()).hexdigest()==pins['notebook_sha256']
    document=json.loads(notebook.read_text());result={}
    for cell in document['cells']:
        if cell['cell_type']!='code':continue
        try:tree=ast.parse(''.join(cell['source']))
        except SyntaxError:continue  # Notebook shell cells are never executed.
        for node in tree.body:
            if isinstance(node,(ast.ClassDef,ast.FunctionDef)):result[node.name]=node
    return result


def notebook_model(root,source,notebook):
    # Reuse independently executed pinned encoder factory, not candidate code.
    original=source_model(root,source)
    encoder_module=original.__init__.__globals__['pretrainedmodels']
    nodes=notebook_nodes(root,notebook)
    names={'conv3x3','conv1x1','init_weight','cSEBlock','sSEBlock','scSEBlock',
           'Attention','ChannelAttentionModule','SpatialAttentionModule','CBAM',
           'CenterBlock','DecodeBlock','UNET_SERESNEXT101'}
    assert names<=nodes.keys()
    ns=dict(torch=torch,np=np,nn=torch.nn,F=torch.nn.functional,
            pretrainedmodels=encoder_module,config=dict(clf_threshold=.5))
    exec(compile(ast.Module(body=[node for name,node in nodes.items() if name in names],type_ignores=[]),
                 '<pinned-final-notebook-network>','exec'),ns)
    return ns['UNET_SERESNEXT101']


def validate(root,source,notebook):
    torch.set_num_threads(1)
    original=notebook_model(root,source,notebook)
    torch.manual_seed(293);candidate=InferenceNetwork((32,32))
    initial_rng=torch.get_rng_state()
    torch.manual_seed(293);reference=original((32,32),False,False,load_weights=False).eval()
    assert torch.equal(initial_rng,torch.get_rng_state())
    exact_state(candidate.state_dict(),reference.state_dict())
    torch.manual_seed(307);images=torch.randn(2,3,32,32)
    cases=[]
    for name,forced,decode in [('natural',None,None),('all_below',[-1.,-2.],False),
                                ('exact_boundary',[0.,0.],False),('mixed',[-1.,1.],True),
                                ('all_above',[1.,2.],True)]:
        handles=[];calls=[0,0];classifier_calls=[0,0]
        for i,model in enumerate([candidate,reference]):
            def count(module,inputs,output,i=i):calls[i]+=1
            handles.append(model.center.register_forward_hook(count))
            def count_classifier(module,inputs,output,i=i):classifier_calls[i]+=1
            handles.append(model.clf.register_forward_hook(count_classifier))
            if forced is not None:
                def override(module,inputs,output):return torch.tensor(forced,dtype=output.dtype).reshape(2,1)
                handles.append(model.clf.register_forward_hook(override))
        try:
            with torch.no_grad():actual=candidate(images);expected=reference(images)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            assert calls[0]==calls[1]
            assert classifier_calls==[1,1+calls[1]]
            if decode is not None:assert calls==[int(decode)]*2
            if decode is False:
                assert torch.count_nonzero(actual)==0
                assert torch.count_nonzero(torch.sigmoid(actual)>.5)==0
            if name=='mixed':assert torch.count_nonzero(actual[0])>0
            cases.append(dict(case=name,decoder_calls=calls[0],classifier_calls=classifier_calls,exact_logits=True))
        finally:
            for handle in handles:handle.remove()
    exact_state(candidate.state_dict(),reference.state_dict())
    try:candidate.train()
    except ValueError:pass
    else:raise AssertionError('Inference-only model accepted training mode')
    paths=['sciona/hubmap_inference_network.py','sciona/hubmap_network.py','sciona/hubmap_encoder.py',
           'scripts/validate_hubmap_final_network.py','scripts/validate_hubmap_network.py',
           'docs/reviews/competition_hubmap_source_pins.json','docs/reviews/competition_hubmap_final_notebook_pins.json']
    return dict(approved=False,synthetic_only=True,parameters=sum(p.numel() for p in candidate.parameters()),
                cases=cases,exact_initial_state_rng_final_buffers=True,
                implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
                limitations=['Final notebook duplicate classifier call verified: two calls after decoding, one on shortcut; eval buffers/logits unchanged.',
                             'Full original encoder/decoder with synthetic initialized weights and CPU float32 inputs.',
                             'Four prescribed classification outputs exercise branch boundaries; natural classifier also compared.',
                             'Plain-output inference only; source invalid auxiliary shortcut placeholders excluded.',
                             'Checkpoint adaptation, complete raw inference and pseudo-label/retraining graph gates remain.'])


if __name__=='__main__':
    signal.alarm(120)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--notebook',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    report=validate(root,args.source_root,args.notebook)
    (root/'docs/reviews/competition_hubmap_final_network.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'implementation_sha256','limitations'}}))
