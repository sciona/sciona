"""Reject stale or reduced M5 Uncertainty implementation evidence before publication."""
import json
import pytest
from scripts import review_decomposed_m5u_implementation as review


@pytest.fixture
def evidence():
    return json.loads((review.ROOT/review.EVIDENCE).read_text())


def test_complete_execution_accepted(evidence):
    assert len(review.validate_execution(evidence).nodes) == 6


@pytest.mark.parametrize('name,value', [
    ('format','wrong'),('status','failed'),('approved',True),('catalog_mutations',1),
    ('synthetic_only',False),('all_nodes_uncached',False)])
def test_headers_rejected(evidence,name,value):
    evidence[name] = value
    with pytest.raises(ValueError,match='headers'): review.validate_execution(evidence)


@pytest.mark.parametrize('name', [
    'actual_runner_nodes','actual_runner_edges','models_fitted','levels','quantiles','horizon','hierarchy_series','candidate_fits','refits','query_rows','full_source_sampling_and_search','outer_group_isolation','exact_reconstruction_result','strict_json_output',
    'graph_codec_roundtrip'])
def test_lifecycle_rejected(evidence,name):
    evidence['checks'][name] = 0
    with pytest.raises(ValueError,match='lifecycle'): review.validate_execution(evidence)


@pytest.mark.parametrize('name,value', [('native_threads',8),('lightgbm_threads',8),('search_workers',8),('serial_models',False)])
def test_resource_policy_rejected(evidence,name,value):
    evidence['resource_policy'][name] = value
    with pytest.raises(ValueError,match='Resource'): review.validate_execution(evidence)


@pytest.mark.parametrize('fault', ['missing','changed','extra'])
def test_code_evidence_rejected(evidence,fault):
    hashes = evidence['code_sha256']
    if fault == 'missing': hashes.pop('sciona/m5u_pipeline.py')
    elif fault == 'changed': hashes['sciona/m5u_pipeline.py'] = '0'*64
    else: hashes['unexpected'] = '0'*64
    with pytest.raises(ValueError,match='Implementation'): review.validate_execution(evidence)


@pytest.mark.parametrize('fault', ['missing','changed','extra'])
def test_provider_evidence_rejected(evidence,fault):
    hashes = evidence['bound_provider_sha256']
    if fault == 'missing': hashes.pop(review.GENERIC)
    elif fault == 'changed': hashes[review.GENERIC] = '0'*64
    else: hashes['unexpected'] = '0'*64
    with pytest.raises(ValueError,match='Provider'): review.validate_execution(evidence)


def test_runtime_drift_rejected(evidence):
    evidence['runtime']['numpy'] = '0.0.0'
    with pytest.raises(ValueError,match='Runtime'): review.validate_execution(evidence)


def test_graph_drift_rejected(evidence):
    evidence['serialized_graph_sha256'] = '0'*64
    with pytest.raises(ValueError,match='Graph'): review.validate_execution(evidence)


def test_missing_full_training_dependency_rejected(evidence,monkeypatch):
    from sciona import decomposed_m5u_runtime as candidate
    monkeypatch.setattr(candidate,'run_unrestored',lambda *args,**kwargs: None)
    with pytest.raises(ValueError,match='dependencies'): review.validate_execution(evidence)


def test_wrong_interface_type_rejected():
    from sciona.atoms.ml.decomposed_m5u_execution import restoration_inputs
    node = next(n for n in review.build_decomposed_m5u_graph().nodes if n.node_id=='restoration_inputs')
    node.outputs[1].type_desc = 'dict'
    with pytest.raises(ValueError,match='Return annotation'): review.interfaces(restoration_inputs,node)
