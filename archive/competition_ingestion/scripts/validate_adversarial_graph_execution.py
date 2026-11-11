"""Real serialized graph, five full models, padded ten-entry twenty-step attack."""
import asyncio
import base64
import gc
import hashlib
from io import BytesIO
import inspect
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import sciona.atoms.dl.adversarial_execution as provider
from sciona.adversarial_graph import build_adversarial_graph
from sciona.adversarial_ensemble import ORDER
from sciona.adversarial_inception_v3 import build_inception_v3
from sciona.adversarial_inception_resnet import build_inception_resnet
from sciona.adversarial_state_mapping import mapping
from sciona.adversarial_image_boundary import source_rgb
from sciona.dfdc_codec import encode_array,encode_state,decode_array
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.visualizer import runner


def run_graph(graph,payload):
    captured={}
    def capture(directory,node,name,value):
        if node=='execute' and name=='out_result':captured['result']=value
    with tempfile.TemporaryDirectory(prefix='sciona-synthetic-adversarial-graph-') as directory:
        with patch.object(runner,'RUNS_DIR',Path(directory)),patch.object(runner,'save_intermediate_value',side_effect=capture):
            status=asyncio.run(runner.CDGExecutionSession(None,'synthetic-adversarial','case').execute({'payload':payload},cdg=graph))
    assert status['status']=='completed'
    return captured['result']


def main():
    torch.set_num_threads(2)
    runner._ensure_atoms_imported()
    rng=torch.get_rng_state().clone();numpy_rng=np.random.get_state()
    specs={}
    for i,scope in enumerate(ORDER['targeted_large']):
        family='inception_resnet' if 'Resnet' in scope else 'inception_v3'
        model=(build_inception_resnet if family=='inception_resnet' else build_inception_v3)(seed=1600+i)
        with torch.no_grad():
            for name,buffer in model.named_buffers():
                if name.endswith('.beta'):buffer.fill_(.1)
        source={};state=model.state_dict()
        for key,(name,layout) in mapping(model,family,scope).items():
            v=state[key]
            source[name]=v.permute(2,3,1,0) if layout=='HWIO' else v.T if layout=='IO' else v
        specs[scope]={'kind':'slim_tensors','seed':1600+i,'tensors':encode_state(source)}
        del model,state,source,v,buffer
        gc.collect()
    rgb=(np.arange(299*299*3,dtype=np.uint32)%256).astype(np.uint8).reshape(1,299,299,3)
    payload={'version':1,'rgb':encode_array(rgb),'targets':encode_array(np.array([17],np.int64)),
             'config':{'mode':'targeted','epsilon':8,'momentum':1.,'non_targeted_iterations':10},
             'initializations':specs}
    digest,nodes,edges=encode_execution_graph(build_adversarial_graph())
    graph=decode_execution_graph(nodes,edges,digest)
    result=run_graph(graph,payload)
    result=json.loads(json.dumps(result,allow_nan=False))
    images=decode_array(result['normalized_images']);labels=decode_array(result['labels'])
    assert images.shape==rgb.shape and labels.tolist()==[17]
    assert len(result['batch_losses'])==1 and len(result['batch_losses'][0])==20
    assert len(result['pngs'])==1
    normalized=(rgb.astype(np.float64)/255.*2.-1.).astype(np.float32)
    assert np.max(np.abs(images-normalized))<=16/255+1e-7 and np.any(images!=normalized)
    decoded=np.array(Image.open(BytesIO(base64.b64decode(result['pngs'][0],validate=True))))
    np.testing.assert_array_equal(decoded,source_rgb(images[0]))
    assert result['initialization_kinds']=={s:'slim_tensors' for s in specs}
    assert torch.equal(rng,torch.get_rng_state())
    after=np.random.get_state()
    assert numpy_rng[0]==after[0] and np.array_equal(numpy_rng[1],after[1]) and numpy_rng[2:]==after[2:]
    paths=[str(p.relative_to(ROOT)) for p in sorted((ROOT/'sciona').glob('adversarial_*.py'))]
    paths += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'sciona').glob('adversarial_*topology.json'))]
    paths += ['scripts/validate_adversarial_graph_execution.py','sciona/dfdc_codec.py']
    report={'format':'adversarial-graph-execution-validation.v1','result':'passed','approved':False,
            'checks':{'actual_runner_nodes':2,'actual_full_models':5,'complete_targeted_large_steps':20,
                      'neural_batch_size':10,'real_entries':1,'padded_entries':9,
                      'encoded_caller_states':True,'normalized_projection_bound':True,
                      'only_real_source_scaled_PNG':True,'post_discovery_rng_unchanged':True},
            'serialized_graph_sha256':digest,
            'provider_sha256':hashlib.sha256(Path(inspect.getfile(provider)).read_bytes()).hexdigest(),
            'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
            'limits':'Actual runner/providers with synthetic supplied states, BNbeta=.1; no pretrained quality or TF binary equivalence. Cold provider discovery occurs before RNG baseline. Intermediate values captured only in memory. No served catalog/publication claim.'}
    (ROOT/'docs/reviews/competition_adversarial_graph_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']),flush=True)


if __name__=='__main__':main()
