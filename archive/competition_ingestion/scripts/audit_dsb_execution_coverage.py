"""Read-only stage coverage audit of the serialized synthetic raw-input graph.

Execution coverage is not independent source equivalence or publication approval.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
from collections import Counter
import pytest
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv
from sciona.competition_graph import load_competition_graph
from sciona.dsb_execution import build_dsb_raw_training_graph
from sciona.services.execution_graph_codec import encode_execution_graph

STAGES={
    'lung_mask_with_bone_removal':['dsb_components:mask_and_remove_bone','dsb_training_preprocessing:prepare_training_masks'],
    'volume_split_combine':['dsb_components:split_volume','dsb_components:combine_volume'],
    'coordinate_aware_3d_unet':['dsb_network:Net.forward','dsb_network:DetectorNet.forward'],
    'anchor_label_mapping_with_iou_dilation':['dsb_components:assign_anchor_labels'],
    'online_hard_negative_mining':['dsb_losses:detector_loss'],
    'size_aware_nodule_oversampling':['dsb_components:oversample_by_size'],
    'softmax_temperature_proposal_sampling':['dsb_components:sample_proposals'],
    'center_feature_extraction_3d':['dsb_network:CaseNet.forward'],
    'noisy_or_pooling':['dsb_network:CaseNet.forward'],
    'miss_penalty_loss':['dsb_losses:classifier_loss'],
}


class Coverage:
    def __init__(self):self.calls=Counter()
    def profile(self,frame,event,arg):
        if event=='call' and '/sciona/dsb_' in frame.f_code.co_filename:
            module=frame.f_globals.get('__name__','').split('.')[-1]
            self.calls[module+':'+frame.f_code.co_qualname]+=1
    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self,item):
        from sciona.visualizer.runner import CDGExecutionSession
        original=CDGExecutionSession.execute
        async def execute(session,*args,**kwargs):
            previous=sys.getprofile();sys.setprofile(self.profile)
            try:return await original(session,*args,**kwargs)
            finally:sys.setprofile(previous)
        # Exclude fixture setup and direct helper calls in the test body. Only
        # the actual serialized graph execution may satisfy stage coverage.
        CDGExecutionSession.execute=execute
        try:yield
        finally:CDGExecutionSession.execute=original


def main():
    signal.alarm(120)
    root=Path(__file__).resolve().parents[1]
    load_dotenv(root/'.env')
    with psycopg.connect(os.environ['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        source=load_competition_graph(db,version_id='6625a41e-b5a7-5ee2-86e5-1d2c60d6d185')
    assert {n.node_id for n in source.nodes}==set(STAGES),'source stage inventory changed'
    coverage=Coverage()
    result=pytest.main(['-q','tests/test_dsb_execution.py::test_serialized_training_checkpoint_reaches_inference[raw]'],plugins=[coverage])
    if result:raise SystemExit(result)
    stages={name:{fn:coverage.calls[fn] for fn in functions} for name,functions in STAGES.items()}
    assert all(count>0 for functions in stages.values() for count in functions.values()),stages
    evidence={};stale=[]
    for path in sorted((root/'docs/reviews').glob('competition_dsb*.json')):
        if path.name=='competition_dsb_execution_coverage.json':continue
        document=json.loads(path.read_text())
        hashes=document.get('implementation_sha256',{})
        if not hashes:continue
        for name,digest in hashes.items():
            if not (root/name).exists() or hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
                stale.append(dict(report=path.name,path=name))
        evidence[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    assert not stale,stale
    graph=build_dsb_raw_training_graph();digest,_,_=encode_execution_graph(graph)
    paths=sorted(root.glob('sciona/dsb_*.py'))+[root/'tests/test_dsb_execution.py',Path(__file__),
        root/'sciona/visualizer/runner.py',root/'sciona/services/execution_graph_codec.py']
    providers=sorted((root.parent/'sciona-atoms-dl/src/sciona/atoms/dl/detection').glob('dsb_*.py'))
    report=dict(approved=False,read_only=True,synthetic_only=True,
        source_version_id='6625a41e-b5a7-5ee2-86e5-1d2c60d6d185',execution_graph_sha256=digest,
        source_stage_count=len(STAGES),executed_stage_implementations=stages,
        evidence_sha256=evidence,stale_evidence=[],
        implementation_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        provider_sha256={str(p.relative_to(root.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in providers},
        limitations=['Call coverage does not establish independent whole-graph source parity.',
            'Center extraction and noisy-OR share the CaseNet forward implementation; their source parity is separate evidence.',
            'Synthetic runtime states and proposal inputs establish no competition accuracy.',
            'Catalog provider, lineage, tier and publication gates remain.'])
    (root/'docs/reviews/competition_dsb_execution_coverage.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'stage_count':len(stages),'stale_evidence':stale,'approved':False}))


if __name__=='__main__':main()
