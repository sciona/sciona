"""Synthetic rejection tests for complete HuBMAP lifecycle evidence."""
from copy import deepcopy
import pytest
from scripts import hubmap_execution_evidence as evidence


def report():
    return dict(format='hubmap-current-graph.v2',passed=True,approved=False,catalog_mutations=0,
        synthetic_only=True,all_nodes_uncached=True,actual_runner_nodes=6,full_training_runs=8,
        source_geometry=dict(tile_size=1024,input_side=320,pad_size=256),synthetic_raster_side=1024,checkpoint_models_per_stage=4,
        generated_nonempty_pseudo_handoff=True,exact_pseudo_state_routing=True,
        resource_policy=deepcopy(evidence.RESOURCE_POLICY),
        folds=[dict(stage=stage,fold=fold,epochs=list(range(1,21)),selected_checkpoint='best_score' if stage=='initial_training' else 'best_loss',optimizer_updates=20,
                    validation_examples=20) for stage in ('initial_training','retraining') for fold in range(4)],
        pseudo_foreground_pixels=10,final_foreground_pixels=0,
        software_sha256={p:'a'*64 for p in evidence.REQUIRED_SOFTWARE},
        graph_sha256='b'*64,prerequisite_sha256='c'*64,limits=['Synthetic source schedule with resource-adapted batch size only.'])


def test_complete_execution():evidence.validate_report(report())


@pytest.mark.parametrize('fault',['missing_fold','duplicate_fold','no_updates','no_validation','wrong_checkpoint',
    'wrong_epoch','old_short_schedule','old_pseudo_selection','partial_models','partial_nodes','partial_training','reduced_geometry','empty_pseudo',
    'handoff','cached','workers','opencv','boolean_count','missing_software','bad_hash','missing_limits','empty_limit'])
def test_reject_incomplete_execution(fault):
    r=report()
    if fault=='missing_fold':r['folds'].pop()
    elif fault=='duplicate_fold':r['folds'][1]['fold']=0
    elif fault=='no_updates':r['folds'][0]['optimizer_updates']=0
    elif fault=='no_validation':r['folds'][4]['validation_examples']=0
    elif fault=='wrong_checkpoint':r['folds'][0]['selected_checkpoint']='snapshot_18'
    elif fault=='wrong_epoch':r['folds'][0]['epochs']=[18]
    elif fault=='old_short_schedule':r['folds'][0]['epochs']=[1,2,3,4]
    elif fault=='old_pseudo_selection':r['folds'][0]['selected_checkpoint']='best_loss'
    elif fault=='partial_models':r['checkpoint_models_per_stage']=1
    elif fault=='partial_nodes':r['actual_runner_nodes']=5
    elif fault=='partial_training':r['full_training_runs']=7
    elif fault=='reduced_geometry':r['source_geometry']['input_side']=32
    elif fault=='empty_pseudo':r['pseudo_foreground_pixels']=0
    elif fault=='handoff':r['exact_pseudo_state_routing']=False
    elif fault=='cached':r['all_nodes_uncached']=False
    elif fault=='workers':r['resource_policy']['data_loader_workers']=4
    elif fault=='opencv':r['resource_policy']['opencv_threads']=16
    elif fault=='boolean_count':r['folds'][0]['optimizer_updates']=True
    elif fault=='missing_software':r['software_sha256'].pop('sciona/hubmap_retraining.py')
    elif fault=='bad_hash':r['prerequisite_sha256']='unverified'
    elif fault=='missing_limits':r.pop('limits')
    else:r['limits']=['']
    with pytest.raises(ValueError):evidence.validate_report(r)


def test_uncommitted_execution_rejected():
    with pytest.raises(ValueError):evidence.review(None)
