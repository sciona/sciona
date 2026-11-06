"""Reject incomplete or stale evidence before provider/catalog publication."""
import copy
import json
from pathlib import Path

import pytest

from scripts import review_decomposed_nab as review


@pytest.fixture
def evidence():
    return json.loads((review.ROOT/review.EVIDENCE).read_text())


def test_current_execution_and_complete_body_pass(evidence):
    review.validate_execution(evidence)
    review.check_forest_loop()


@pytest.mark.parametrize('field,value', [('format', 'unknown'), ('status', 'failed'), ('approved', True)])
def test_execution_identity_rejected(evidence, field, value):
    evidence[field] = value
    with pytest.raises(ValueError, match='execution evidence'):
        review.validate_execution(evidence)


@pytest.mark.parametrize('field,value', [('actual_runner_nodes', 5), ('actual_runner_edges', 12),
    ('models_fitted', 89), ('stream_points', 99), ('exact_reconstruction_result', False),
    ('strict_json_output', False), ('graph_codec_roundtrip', False)])
def test_incomplete_lifecycle_rejected(evidence, field, value):
    evidence['checks'][field] = value
    with pytest.raises(ValueError, match='execution coverage'):
        review.validate_execution(evidence)


@pytest.mark.parametrize('field,value', [('native_threads', 8), ('forest_jobs', 4),
                                       ('serial_models', False)])
def test_resource_policy_rejected(evidence, field, value):
    evidence['resource_policy'][field] = value
    with pytest.raises(ValueError, match='resource policy'):
        review.validate_execution(evidence)


def test_incomplete_hash_coverage_rejected(evidence):
    evidence['code_sha256'].pop('sciona/decomposed_nab_runtime.py')
    with pytest.raises(ValueError, match='hash coverage'):
        review.validate_execution(evidence)


def test_changed_implementation_rejected(evidence):
    evidence['code_sha256']['sciona/decomposed_nab_runtime.py'] = '0' * 64
    with pytest.raises(ValueError, match='Implementation changed'):
        review.validate_execution(evidence)


def test_incomplete_package_coverage_rejected(evidence):
    evidence['runtime'].pop('numpy')
    with pytest.raises(ValueError, match='runtime versions'):
        review.validate_execution(evidence)


def test_changed_package_version_rejected(evidence):
    evidence['runtime']['numpy'] = '0.0.0'
    with pytest.raises(ValueError, match='Runtime version changed'):
        review.validate_execution(evidence)


def test_changed_training_dependency_rejected(monkeypatch):
    from sciona import decomposed_nab_runtime as candidate
    monkeypatch.setattr(candidate, 'IsolationForest', lambda *a, **kw: None)
    with pytest.raises(ValueError, match='Forest dependency changed'):
        review.check_forest_loop()


def test_rewritten_execution_report_rejected(monkeypatch):
    read = Path.read_bytes
    monkeypatch.setattr(Path, 'read_bytes', lambda path: b'{}' if path == review.ROOT/review.EVIDENCE else read(path))
    with pytest.raises(ValueError, match='Immutable execution evidence'):
        review.review()


@pytest.mark.parametrize('fault', ['input_order', 'input_type', 'required', 'output_type'])
def test_changed_provider_interface_rejected(fault):
    from sciona.atoms.numerical.statistics.prior_threshold import prior_location_scale_threshold
    graph = review.build_decomposed_nab_graph()
    node = copy.deepcopy(graph.nodes[3])
    if fault == 'input_order':
        node.inputs.reverse()
    elif fault == 'input_type':
        node.inputs[0].type_desc = 'dict'
    elif fault == 'required':
        node.inputs[0].required = False
    else:
        node.outputs[0].type_desc = 'dict'
    with pytest.raises(ValueError):
        review.interfaces(prior_location_scale_threshold, [node])


def test_altered_forest_loop_rejected(monkeypatch):
    original = review.inspect.getsource
    def changed(function):
        source = original(function)
        if function.__name__ == 'forest_scores':
            source = source.replace('n_jobs=1', 'n_jobs=4')
        return source
    monkeypatch.setattr(review.inspect, 'getsource', changed)
    with pytest.raises(ValueError, match='forest loop changed'):
        review.check_forest_loop()
