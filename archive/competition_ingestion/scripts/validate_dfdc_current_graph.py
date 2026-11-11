"""Full current serialized DFDC graph with synthetic native and neural execution."""
import asyncio
import cv2
from threadpoolctl import threadpool_limits
import subprocess
import json
from pathlib import Path
import random
import sys
import tempfile
from unittest.mock import patch
import warnings

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.dfdc_codec import encode_array,encode_state
from sciona.dfdc_detector import build_detector
from sciona.dfdc_graph import build_dfdc_graph
from sciona.services.execution_graph_codec import encode_execution_graph, decode_execution_graph
from sciona.visualizer import runner
from scripts.audit_dfdc_original_readiness import sha
from scripts.review_dfdc_full_runtime import review as runtime_review
from scripts.validate_dfdc_hull_backend import train_synthetic
from scripts.validate_dfdc_training import assert_equal



def prerequisites():
    reports={}
    for name,commit in [('dfdc_current_contracts','43fdbeea'),('dfdc_full_runtime','858ad8ea')]:
        path='docs/reviews/'+name+'.json'
        raw=(ROOT/path).read_bytes()
        if raw!=subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT):
            raise ValueError('Immutable prerequisite changed')
        report=json.loads(raw)
        if report['passed'] is not True:
            raise ValueError('Passing prerequisites required')
        for path,digest in report['software_sha256'].items():
            if sha(ROOT/path)!=digest:
                raise ValueError('Qualified software changed')
        reports[name]=report
    if reports['dfdc_current_contracts']['tests']!=dict(tests=76,failures=0,errors=0,skipped=0):
        raise ValueError('Complete component tests required')
    if runtime_review()!=reports['dfdc_full_runtime']:
        raise ValueError('Qualified runtime changed')
    return reports


def execute(payload):
    digest,nodes,edges=encode_execution_graph(build_dfdc_graph())
    graph=decode_execution_graph(nodes,edges,digest)
    if graph!=build_dfdc_graph():
        raise ValueError('Graph roundtrip differs')
    captured={}
    def capture(directory,node,name,value):
        if (node,name) in [('prepare','out_prepared'),('execute','in_prepared'),('execute','out_result')]:
            captured[node,name]=value
    with tempfile.TemporaryDirectory(prefix='synthetic-dfdc-current-') as temporary, threadpool_limits(limits=1):
        with patch.object(runner,'RUNS_DIR',Path(temporary)),patch.object(runner,'save_intermediate_value',side_effect=capture):
            result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-dfdc','current').execute({'payload':payload},cdg=graph))
    if result['status']!='completed':
        raise ValueError('Full graph execution did not complete')
    if len(result['trace'])!=2 or {r['node_id'] for r in result['trace']}!={'prepare','execute'} or any(r['cached'] for r in result['trace']):
        raise ValueError('Both nodes must execute uncached')
    if captured['prepare','out_prepared'] is not captured['execute','in_prepared']:
        raise ValueError('Prepared state routing differs')
    return captured['execute','out_result']


def main():
    baseline=prerequisites()
    paths=sorted(set(baseline['dfdc_current_contracts']['software_sha256']) | {
        'scripts/validate_dfdc_current_graph.py','scripts/validate_dfdc_hull_backend.py',
        'scripts/validate_dfdc_training.py','sciona/services/execution_graph_codec.py','sciona/visualizer/runner.py'})
    before_hashes={p:sha(ROOT/p) for p in paths}
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    if cv2.getNumThreads()!=1 or cv2.ocl.useOpenCL():
        raise ValueError('Bounded OpenCV required')
    runner._ensure_atoms_imported()
    detector=build_detector(initialization='random',seed=327)
    with torch.no_grad():
        for parameter in detector.parameters():parameter.zero_()
        for model,head in ((detector.pnet,'conv4_1'),(detector.rnet,'dense5_1'),(detector.onet,'dense6_1')):
            getattr(model,head).bias.copy_(torch.tensor([-2.,2.]))
        detector.onet.dense6_3.bias.fill_(.5)
    state=encode_state(detector.state_dict());del detector
    original=np.random.default_rng(328).integers(1,256,(11,80,80,3),dtype=np.uint8)
    frames=[original,255-original,original.copy(),255-original]
    training=[{'frames':encode_array(frame),'part':part,'original':parent}
              for frame,part,parent in zip(frames,(0,0,25,25),(0,0,2,2))]
    predictor,_,_,_=train_synthetic()
    with tempfile.TemporaryDirectory(prefix='sciona-synthetic-runtime-') as directory:
        native=Path(directory)/'predictor.dat';predictor.save(str(native))
        payload={'version':1,'training':training,'prediction':[encode_array(original[:2])],
                 'detector':{'policy':'state','seed':327,'state':state},
                 'classifier':{'policy':'random'},'hull':{'predictor_path':str(native)},
                 'plan':{'runs':[[seed,0,2] for seed in (111,555,777,888,999)],
                         'requests':[[111,0,0],[555,0,0],[777,0,0],[777,0,1],[888,0,0],[888,0,1],[999,0,0]]},
                 'config':{'record_order_seed':329,'n_splits':2,'batch_size':1,'batches_per_epoch':1,
                           'frames_per_video':2,'precision':'float32'}}
        payload=json.loads(json.dumps(payload,allow_nan=False))
        before=(random.getstate(),np.random.get_state(),torch.get_rng_state())
        with warnings.catch_warnings(),patch('torch.load',side_effect=AssertionError('implicit pickle load')):
            warnings.simplefilter('ignore',UserWarning)
            result=execute(payload)
        assert_equal(before,(random.getstate(),np.random.get_state(),torch.get_rng_state()))
        assert len(result['runs'])==5 and result['training_records']>0
        assert [(r['seed'],r['fold']) for r in result['runs']]==[(seed,0) for seed in (111,555,777,888,999)]
        assert all(len(r['training']['epochs'])==2 for r in result['runs'])
        assert result['plan']['runs']==payload['plan']['runs'] and result['plan']['requests']==payload['plan']['requests']
        assert sum(e['training']['optimizer_updates'] for run in result['runs'] for e in run['training']['epochs'])==10
        assert all(e['validation']['clips']==2 for run in result['runs'] for e in run['training']['epochs'])
        prediction=result['predictions'][0]
        assert prediction['models']==7 and prediction['status']=='predicted'
        assert prediction['classified_faces']>=2 and np.isfinite(prediction['score'])
        assert not result['plan']['source_requests_preserved'] and not result['plan']['source_runs_preserved']
        serialized=json.dumps(result,allow_nan=False)
        assert str(native) not in serialized and 'predictor_path' not in serialized and 'state_dict' not in serialized
    if prerequisites()!=baseline or before_hashes!={p:sha(ROOT/p) for p in paths}:
        raise ValueError('Runtime or software changed during execution')
    report={'format':'dfdc-current-graph.v1','approved':False,'catalog_mutations':0,'synthetic_only':True,
            'serialized_graph_sha256':encode_execution_graph(build_dfdc_graph())[0],
            'resource_policy':dict(serial_models=True,torch_threads=1,interop_threads=1,native_threads=1,opencv_threads=1,opencv_opencl=False,data_loader_workers=0),
            'all_nodes_uncached':True,'exact_state_routing':True,'result':'passed',
            'checks':{'json_input_roundtrip':True,'actual_MTCNN_preparation_and_prediction':True,
                      'actual_native_synthetic_predictor_loaded':True,'full_B7_training_runs':5,
                      'connected_training_updates':10,'validation_epochs':10,'selected_ensemble_models':7,
                      'nonfallback_prediction':True,'caller_rng_restored':True,'runtime_path_and_states_excluded_from_output':True},
            'aggregate':{'training_records':result['training_records'],'classified_faces':prediction['classified_faces']},
            'limits':'All runtime stages executed without stand-ins. Detector uses constructed synthetic state; dlib68-point predictor trained only on synthetic images. Native loader validates predictor but positive HOG face quality is not claimed. Five B7 runs use random initialization, two epochs/one batch1 update each,380 CPUfloat32. Explicit shortened request plan, not original checkpoint provenance or default-scale quality. Current serialized graph/providers executed; corrected stored original and publication remain separate.',
            'software_sha256':before_hashes,
            'prerequisite_sha256':{name:sha(ROOT/('docs/reviews/'+name+'.json')) for name in baseline}}
    (ROOT/'docs/reviews/dfdc_current_graph.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
