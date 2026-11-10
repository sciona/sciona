"""Execute all four configured HuBMAP folds at source geometry, with synthetic data."""
import asyncio
import gc
import json
import random
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from scripts.audit_hubmap_original_readiness import ROOT, sha
from scripts.hubmap_structured_synthetic_rasters import labeled_raster
from scripts import review_video_numeric_runtime as runtime_checker
from sciona.hubmap_encoder import build_encoder
from sciona.hubmap_lifecycle import FoldStage
from sciona.hubmap_execution import build_hubmap_lifecycle_graph
from sciona.services.execution_graph_codec import encode_execution_graph,decode_execution_graph
from sciona.visualizer import runner


def prerequisites():
    path='docs/reviews/hubmap_current_contracts.json'
    raw=(ROOT/path).read_bytes()
    if raw!=subprocess.check_output(['git','show','993e042d:'+path],cwd=ROOT):
        raise ValueError('Immutable component qualification changed')
    report=json.loads(raw)
    if report['passed'] is not True or report['tests']!=dict(tests=49,failures=0,errors=0,skipped=0):
        raise ValueError('Complete component qualification required')
    for name,digest in report['software_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Qualified software changed')
    previous=runtime_checker.ROOT_PACKAGES
    runtime_checker.ROOT_PACKAGES=report['runtime']['root_packages']
    try:
        if runtime_checker.review()!=report['runtime']:raise ValueError('Qualified runtime changed')
    finally:runtime_checker.ROOT_PACKAGES=previous
    return report


def main():
    baseline=prerequisites()
    policy_path='docs/reviews/hubmap_source_policy.json'
    policy_raw=subprocess.check_output(['git','show','90963b28:'+policy_path],cwd=ROOT)
    if policy_raw!=(ROOT/policy_path).read_bytes():raise ValueError('Immutable source policy changed')
    policy=json.loads(policy_raw)
    if (policy['reviewer_sha256']!=sha(ROOT/'scripts/review_hubmap_source_policy.py')
            or policy['pseudo_checkpoint_role']!='best_score'
            or any(policy[k]['num_epochs']!=20 or policy[k]['patience']!=50 for k in ('initial_training','retraining'))):
        raise ValueError('Qualified source schedule and pseudo checkpoint selection required')
    paths=set(baseline['software_sha256'])|{'scripts/validate_hubmap_current_graph.py','scripts/hubmap_structured_synthetic_rasters.py',
        'sciona/services/execution_graph_codec.py','sciona/visualizer/runner.py',
        policy_path,'scripts/review_hubmap_source_policy.py'}
    hashes={p:sha(ROOT/p) for p in sorted(paths)}
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    cv2.setNumThreads(0);cv2.ocl.setUseOpenCL(False)
    if cv2.getNumThreads()!=1 or cv2.ocl.useOpenCL():raise ValueError('Bounded OpenCV required')
    runner._ensure_atoms_imported()
    torch.manual_seed(367)
    encoder=build_encoder();state=encoder.state_dict();del encoder
    slides=[];groups=[]
    for fold in range(4):
        for foreground in (False,True):
            image,mask=labeled_raster(410+fold*2+int(foreground),foreground,side=1024)
            slides.append((image,mask));groups.append('synthetic-group-'+str(fold))
    def stages(offset):
        return [FoldStage(state,offset+fold,['synthetic-group-'+str(fold)],
            dict(start_epoch=1,end_epoch=20,patience=50,maximum_bin=4,training_batch_size=2,
                 validation_batch_size=2,input_side=320,python_rng=random.Random(offset+fold),
                 numpy_rng=np.random.RandomState(offset+fold+1),
                 torch_generator=torch.Generator().manual_seed(offset+fold+2))) for fold in range(4)]
    pseudo,_=labeled_raster(479,True,side=1024)
    final,_=labeled_raster(483,True,side=1024)
    inputs=dict(slides=slides,groups=groups,pseudo_sources=[[pseudo]],final_images=[final],
        initial_stages=stages(0),retraining_stages=stages(100),
        initial_selectors=[(i,'best_score') for i in range(4)],final_selectors=[(i,'best_loss') for i in range(4)],
        tile_size=1024,input_resolution=320,
        pseudo_rng=dict(python_rng=random.Random(489),torch_generator=torch.Generator().manual_seed(497)),
        final_rng=dict(python_rng=random.Random(501),torch_generator=torch.Generator().manual_seed(509)),
        inference_geometry=dict(resolution=1024,input_resolution=320,pad_size=256,batch_size=2))
    graph=build_hubmap_lifecycle_graph();digest,nodes,edges=encode_execution_graph(graph)
    decoded=decode_execution_graph(nodes,edges,digest)
    if decoded!=graph:raise ValueError('Graph roundtrip differs')
    metrics={};counts={};routing={};prediction={}
    def capture(directory,node,name,value):
        if name in ('out_initial_metrics','out_retraining_metrics'):metrics[node]=value
        if name in ('out_initial_checkpoints','out_final_checkpoints','out_models'):counts[node]=len(value)
        if (node,name)==('pseudo_labels','out_pseudo_labeled_sources'):
            routing['pseudo']=value
            counts['pseudo_foreground_pixels']=sum(int(mask.sum()) for source in value for _,mask in source)
            print('Generated pseudo foreground pixels: '+str(counts['pseudo_foreground_pixels']),flush=True)
            if counts['pseudo_foreground_pixels']==0:raise ValueError('Generated pseudo masks are empty; qualification stops before retraining')
        if (node,name)==('retraining','in_pseudo_labeled_sources'):
            if value is not routing['pseudo']:raise ValueError('Pseudo handoff identity differs')
            routing['verified']=True
        if (node,name)==('final_prediction','out_predictions'):
            if len(value)!=1 or value[0].shape!=(1024,1024) or value[0].dtype!=np.uint8 or not np.isin(value[0],[0,1]).all():
                raise ValueError('Complete binary raster prediction required')
            prediction['foreground_pixels']=int(value[0].sum())
        if name.startswith('out_'):print('Completed '+node,flush=True)
    with tempfile.TemporaryDirectory(prefix='synthetic-hubmap-current-') as directory,threadpool_limits(limits=1):
        with patch.object(runner,'RUNS_DIR',Path(directory)),patch.object(runner,'save_intermediate_value',side_effect=capture):
            result=asyncio.run(runner.CDGExecutionSession(None,'synthetic-hubmap','current').execute(inputs,cdg=decoded))
    if result['status']!='completed':raise ValueError('Complete lifecycle execution required')
    if len(result['trace'])!=6 or any(r['cached'] for r in result['trace']):raise ValueError('All six nodes must execute uncached')
    if not routing.get('verified'):raise ValueError('Exact pseudo handoff routing required')
    if counts.get('pseudo_foreground_pixels',0)<=0:raise ValueError('Nonempty generated pseudo masks required')
    for node in ('initial_training','initial_models','retraining','final_models'):
        if counts.get(node)!=4:raise ValueError('Every configured fold checkpoint/model required')
    summaries=[]
    for stage in ('initial_training','retraining'):
        if len(metrics[stage])!=4:raise ValueError('Four trained folds required')
        for fold,epochs in enumerate(metrics[stage]):
            if [e['epoch'] for e in epochs]!=list(range(1,21)):raise ValueError('Complete twenty-epoch source schedule required')
            if any(e['training']['examples']<2 or e['validation']['examples']<1 for e in epochs):raise ValueError('Connected training and validation required')
            summaries.append(dict(stage=stage,fold=fold,epochs=list(range(1,21)),
                                  optimizer_updates=sum(e['training']['examples']//2 for e in epochs),
                                  validation_examples=sum(e['validation']['examples'] for e in epochs),selected_checkpoint='best_score' if stage=='initial_training' else 'best_loss'))
    routing.clear();gc.collect()
    if prerequisites()!=baseline or hashes!={p:sha(ROOT/p) for p in hashes}:raise ValueError('Execution software changed')
    report=dict(format='hubmap-current-graph.v2',passed=True,approved=False,catalog_mutations=0,synthetic_only=True,
        graph_sha256=digest,all_nodes_uncached=True,actual_runner_nodes=6,full_training_runs=8,
        source_geometry=dict(tile_size=1024,input_side=320,pad_size=256),synthetic_raster_side=1024,folds=summaries,
        checkpoint_models_per_stage=4,generated_nonempty_pseudo_handoff=True,exact_pseudo_state_routing=True,
        pseudo_foreground_pixels=counts['pseudo_foreground_pixels'],final_foreground_pixels=prediction['foreground_pixels'],
        resource_policy=dict(serial_training=True,torch_threads=1,interop_threads=1,native_threads=1,
                             opencv_threads=1,opencv_opencl=False,data_loader_workers=0),
        software_sha256=hashes,prerequisite_sha256=sha(ROOT/'docs/reviews/hubmap_current_contracts.json'),
        limits=['Synthetic random encoder state, four initial and four fresh retraining folds, twenty consecutive epochs from epoch1 per fold with source patience50 and resource-adapted batch2. No historical trained-weight or predictive accuracy claim.',
                'Source-sized network and tile geometry with synthetic rasters; no whole-slide codec, GPU or unbounded-memory qualification.',
                'Four initial best-score checkpoints follow the pinned pseudo policy. Four final best-loss checkpoints are an explicit caller selection; historical final notebook checkpoint policy and provenance remain unverified.',
                'Synthetic circular foreground is encoded by raster intensity; no real data or application accuracy evaluation is involved.',
                'Existing source-comparison evidence is retained separately; this run is not a fresh independent-source parity comparison.',
                'Corrected stored original execution and publication remain separate requirements.'])
    (ROOT/'docs/reviews/hubmap_current_graph.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,full_training_runs=8,optimizer_updates=sum(x['optimizer_updates'] for x in summaries))))


if __name__=='__main__':main()
