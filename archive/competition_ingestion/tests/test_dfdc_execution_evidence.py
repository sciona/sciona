"""Synthetic rejection cases for DFDC complete-graph evidence."""
from copy import deepcopy
import pytest
from scripts import dfdc_execution_evidence as evidence


def report():
    return dict(format='dfdc-current-graph.v1',result='passed',approved=False,catalog_mutations=0,
        synthetic_only=True,all_nodes_uncached=True,exact_state_routing=True,
        checks=deepcopy(evidence.CHECKS),resource_policy=deepcopy(evidence.RESOURCE_POLICY),
        aggregate=dict(training_records=4,classified_faces=2),
        software_sha256={name:'a'*64 for name in evidence.REQUIRED_SOFTWARE},
        prerequisite_sha256=dict(dfdc_current_contracts='b'*64,dfdc_full_runtime='c'*64),
        limits='Synthetic execution with explicit shortened plan.')


def test_complete_graph():
    evidence.validate_report(report())


@pytest.mark.parametrize('fault',['partial_runs','partial_updates','partial_validation','partial_ensemble',
    'fallback','detector','native_predictor','rng','cached','routing','opencv','workers',
    'boolean_mutation','missing_software','missing_prerequisite','empty_limits','single_face'])
def test_reject_incomplete_execution(fault):
    r=report()
    checks={'partial_runs':('full_B7_training_runs',4),'partial_updates':('connected_training_updates',9),
        'partial_validation':('validation_epochs',9),'partial_ensemble':('selected_ensemble_models',6),
        'fallback':('nonfallback_prediction',False),'detector':('actual_MTCNN_preparation_and_prediction',False),
        'native_predictor':('actual_native_synthetic_predictor_loaded',False),'rng':('caller_rng_restored',False)}
    if fault in checks:
        key,value=checks[fault];r['checks'][key]=value
    elif fault=='cached':r['all_nodes_uncached']=False
    elif fault=='routing':r['exact_state_routing']=False
    elif fault=='opencv':r['resource_policy']['opencv_threads']=16
    elif fault=='workers':r['resource_policy']['data_loader_workers']=4
    elif fault=='boolean_mutation':r['catalog_mutations']=False
    elif fault=='missing_software':r['software_sha256'].pop('sciona/dfdc_selection.py')
    elif fault=='missing_prerequisite':r['prerequisite_sha256'].pop('dfdc_full_runtime')
    elif fault=='empty_limits':r['limits']=''
    else:r['aggregate']['classified_faces']=1
    with pytest.raises(ValueError):evidence.validate_report(r)


def test_uncommitted_execution_rejected():
    with pytest.raises(ValueError):evidence.review(None)
