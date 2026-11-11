"""Synthetic rejection cases for incomplete stored Cornell qualification."""
from copy import deepcopy
import pytest
from scripts import cornell_publication_evidence as evidence
from scripts.review_cornell_original_scope import STAGES


def reports():
    execution = dict(passed=True, synthetic_only=True, stored_graph_verified_before_and_after=True,
        all_nodes_uncached=True, exact_state_routing=True, full_thirty_second_training=True,
        ten_copy_inference=True, strict_json_output=True, models_completed=13,
        training_phases=17, continuations=4, actual_runner_nodes=2, catalog_mutations=0,
        phases=[dict(epochs=2, optimizer_steps=4 if 4 <= i < 9 else 2,
                     mixup=i >= 9, selected_epoch=2) for i in range(17)],
        resource_policy=dict(serial_models=True, torch_threads=1, interop_threads=1, native_threads=1),
        software_sha256={'synthetic_software.py':'0'*64})
    return execution, dict(passed=True, source_stage_review=[dict(stage_id=s) for s in STAGES])


def test_complete_synthetic_evidence():
    execution, source = reports()
    evidence.validate_reports(execution, source, deepcopy(execution), deepcopy(source))


@pytest.mark.parametrize('field,value', [
    ('passed', 1), ('catalog_mutations', False), ('stored_graph_verified_before_and_after', False),
    ('models_completed', 12), ('training_phases', 13), ('continuations', 0),
    ('all_nodes_uncached', False), ('exact_state_routing', False),
    ('full_thirty_second_training', False), ('ten_copy_inference', False),
    ('strict_json_output', False), ('software_sha256', {}), ('resource_policy', {}),
])
def test_reject_incomplete_even_if_expected_matches(field, value):
    execution, source = reports()
    execution[field] = value
    with pytest.raises(ValueError):
        evidence.validate_reports(execution, source, deepcopy(execution), deepcopy(source))


@pytest.mark.parametrize('mutation', ['phase_missing', 'phase_shortened', 'no_mixup', 'wrong_checkpoint', 'stage_missing'])
def test_reject_partial_lifecycle_or_scope(mutation):
    execution, source = reports()
    if mutation == 'phase_missing': execution['phases'].pop()
    elif mutation == 'phase_shortened': execution['phases'][0]['epochs'] = 1
    elif mutation == 'no_mixup': execution['phases'][9]['mixup'] = False
    elif mutation == 'wrong_checkpoint': execution['phases'][0]['selected_epoch'] = 1
    else: source['source_stage_review'].pop()
    with pytest.raises(ValueError):
        evidence.validate_reports(execution, source, deepcopy(execution), deepcopy(source))


def test_reject_changed_immutable_report():
    execution, source = reports()
    expected = deepcopy(execution)
    execution['software_sha256']['synthetic_software.py'] = '1'*64
    with pytest.raises(ValueError):
        evidence.validate_reports(execution, source, expected, deepcopy(source))


def test_reject_missing_execution_commit(monkeypatch):
    monkeypatch.setattr(evidence, 'EVIDENCE_COMMIT', None)
    with pytest.raises(ValueError, match='Committed complete stored execution'):
        evidence.committed_report(evidence.EXECUTION)
