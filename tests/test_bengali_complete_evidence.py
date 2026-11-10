"""Reject incomplete graph qualification independently of a live training run."""
from copy import deepcopy
import pytest
from scripts import bengali_complete_evidence as evidence


def report():
    return dict(format='bengali-complete-graph.v1', passed=True, approved=False,
        catalog_mutations=0, synthetic_only=True, actual_runner_nodes=22,
        all_nodes_uncached=True, exact_edge_handoffs=32, training=deepcopy(evidence.TRAINING),
        inference_checkpoint_replays=8, independent_calibration_scores_match=True,
        final_routing_and_components_match=True, known_training_classes=2, font_only_classes=1,
        calibration_examples=3, query_examples=3, query_alternate_routes=0, fitted_threshold=0.,
        resource_policy=dict(evidence.RESOURCE_POLICY), graph_sha256='a'*64,
        implementation_sha256={name:'b'*64 for name in evidence.REQUIRED_SOFTWARE},
        prerequisite_sha256={name:'c'*64 for name in (
            'docs/reviews/bengali_b7_lifecycle.json', 'docs/reviews/bengali_paired_branches.json')},
        limits=['Synthetic contract fixture; no empirical transfer or approval claim.'])


def test_complete_contract_allows_zero_alternate_routes():
    # Training need not make a useful predictor to establish execution evidence.
    evidence.validate_report(report())


@pytest.mark.parametrize('branch', ['train_ood', 'train_seen'])
@pytest.mark.parametrize('epoch', [198, 200])
def test_last_checkpoint_uses_zero_based_final_epoch(branch, epoch):
    value = report(); value['training'][branch]['selected_epoch'] = epoch
    with pytest.raises(ValueError, match='training'):
        evidence.validate_report(value)


@pytest.mark.parametrize('field,value', [
    ('actual_runner_nodes', 21), ('exact_edge_handoffs', 31),
    ('all_nodes_uncached', False), ('inference_checkpoint_replays', 7),
    ('independent_calibration_scores_match', False), ('final_routing_and_components_match', False),
    ('font_only_classes', 0), ('known_training_classes', True), ('catalog_mutations', False),
    ('approved', True), ('synthetic_only', False), ('query_alternate_routes', True),
    ('query_alternate_routes', -1), ('query_alternate_routes', 4),
    ('fitted_threshold', float('nan')), ('fitted_threshold', float('inf')),
    ('fitted_threshold', True), ('fitted_threshold', 1.01), ('limits', []),
    ('limits', ['']), ('graph_sha256', 'not-a-hash'),
])
def test_reject_incomplete_or_malformed_report(field, value):
    value_report = report(); value_report[field] = value
    with pytest.raises(ValueError):
        evidence.validate_report(value_report)


@pytest.mark.parametrize('branch', list(evidence.TRAINING))
@pytest.mark.parametrize('fault', ['missing', 'short_updates', 'short_epochs'])
def test_every_training_branch_required(branch, fault):
    value = report()
    if fault == 'missing':
        del value['training'][branch]
    else:
        field = 'updates' if fault == 'short_updates' else 'epochs'
        value['training'][branch][field] -= 1
    with pytest.raises(ValueError, match='training'):
        evidence.validate_report(value)


@pytest.mark.parametrize('branch,field', [
    ('train_ood', 'backbone_updated'), ('train_seen', 'backbone_updated'),
    ('train_ood', 'selected_epoch'), ('train_seen', 'selected_epoch'),
    ('train_generator_a', 'all_four_networks_updated'),
    ('train_generator_b', 'all_four_networks_updated'),
    ('train_generator_a', 'frozen_classifier_unchanged'),
    ('train_generator_b', 'frozen_classifier_unchanged'),
])
def test_checkpoint_and_network_state_claims_required(branch, field):
    value = report(); del value['training'][branch][field]
    with pytest.raises(ValueError, match='training'):
        evidence.validate_report(value)


@pytest.mark.parametrize('field', ['implementation_sha256', 'prerequisite_sha256'])
def test_missing_software_or_prerequisite_rejected(field):
    value = report(); value[field].pop(next(iter(value[field])))
    with pytest.raises(ValueError):
        evidence.validate_report(value)


@pytest.mark.parametrize('commit', [None, '', 'a'*8, 'g'*40, True])
def test_uncommitted_evidence_never_reaches_external_state(monkeypatch, commit):
    def forbidden(*args, **kwargs):
        pytest.fail('Invalid commit reached external state')
    monkeypatch.setattr(evidence.subprocess, 'check_output', forbidden)
    with pytest.raises(ValueError, match='Full immutable execution commit'):
        evidence.review(commit)
