import numpy as np
import pytest
import torch

from sciona.grapheme_classifier_inference import classifier_predictions


def test_independent_sigmoid_confidence_and_explicit_softmax_decoder():
    torch.set_num_threads(1)
    ood = torch.full((2, 1295), -20.)
    ood[0, 19] = 0.
    ood[1, 27] = 20.
    seen = torch.full((2, 14784), -20.)
    seen[0, 123] = 20.
    seen[1, 14783] = 20.
    result = classifier_predictions(ood, seen)
    np.testing.assert_array_equal(result['ood_confidence'], [.5, 1.])
    np.testing.assert_array_equal(result['seen_joint_classes'], [123, 14783])
    assert result['ood_confidence'].dtype == np.float64
    assert result['seen_joint_classes'].dtype == np.int64


def test_softmax_rounding_and_tie_policy_are_not_silently_replaced_by_logit_argmax():
    ood = torch.zeros(1, 1295)
    seen = torch.zeros(1, 14784)
    seen[0, 1] = 1e-8
    assert seen.argmax(1).item() == 1
    assert seen.softmax(1)[0, 0] == seen.softmax(1)[0, 1]
    assert classifier_predictions(ood, seen)['seen_joint_classes'].tolist() == [0]


@pytest.mark.parametrize('fault', ['ood_width', 'seen_width', 'rows', 'empty', 'dtype', 'nan', 'inf'])
def test_bad_classifier_boundaries_rejected(fault):
    ood, seen = torch.zeros(2, 1295), torch.zeros(2, 14784)
    if fault == 'ood_width': ood = ood[:, :-1]
    elif fault == 'seen_width': seen = seen[:, :-1]
    elif fault == 'rows': seen = seen[:1]
    elif fault == 'empty': ood, seen = ood[:0], seen[:0]
    elif fault == 'dtype': ood = ood.double()
    elif fault == 'nan': ood[0, 0] = float('nan')
    else: seen[0, 0] = float('inf')
    with pytest.raises(ValueError): classifier_predictions(ood, seen)
