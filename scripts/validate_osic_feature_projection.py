"""Synthetic qualification of the frozen-feature output port adapter."""
import hashlib
import inspect
import itertools
import json
from pathlib import Path

from sciona.atoms.ml.domain_adapters import osic_feature_projection as provider

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('features', 'feature_columns', 'observed_mask', 'observed_values')


def validate():
    # Deliberately arbitrary objects: this adapter must not inspect numerical
    # values or claim to validate the upstream feature-state contract.
    values = (object(), [['synthetic', 'coordinate']], [False, True], [0, 2])
    permutations = 0
    for order in itertools.permutations(range(4)):
        prepared = {NAMES[i]: values[i] for i in order}
        before = list(prepared.items())
        result = provider.unpack_quantile_features(prepared)
        assert type(result) is tuple and len(result) == 4
        assert all(actual is expected for actual, expected in zip(result, values))
        assert list(prepared) == [key for key, _ in before]
        assert all(prepared[key] is value for key, value in before)
        permutations += 1

    shared = []
    mapping = dict(zip(NAMES, (shared, shared, None, shared)))
    result = provider.unpack_quantile_features(mapping)
    assert result[0] is result[1] is result[3] is shared and result[2] is None
    result[0].append('synthetic mutation')
    assert mapping['feature_columns'] == ['synthetic mutation']
    mapping['features'] = object()
    assert result[0] is shared  # Outer projection is independent of key replacement.

    class DictSubclass(dict):
        pass

    invalid = [('not_mapping', []), ('mapping_subclass', DictSubclass(zip(NAMES, values))),
               ('empty_mapping', {}), ('extra_field', dict(zip(NAMES, values), extra=None))]
    for missing in NAMES:
        invalid.append(('missing_' + missing, {k: v for k, v in zip(NAMES, values) if k != missing}))
    renamed = dict(zip(NAMES, values))
    renamed[' features '] = renamed.pop('features')
    invalid.append(('no_implicit_key_trim', renamed))
    rejected = []
    for label, prepared in invalid:
        try:
            provider.unpack_quantile_features(prepared)
        except ValueError:
            rejected.append(label)
        else:
            raise AssertionError('Invalid projection accepted: ' + label)
    try:
        provider.witness_projection()
    except ValueError:
        rejected.append('abstract_execution')
    else:
        raise AssertionError('Abstract projection accepted')

    return dict(passed=True, approved=False, synthetic_only=True, catalog_mutations=0,
        insertion_orders_checked=permutations, fixed_output_order=list(NAMES),
        borrowed_identity_preserved=True, input_mapping_unchanged=True,
        shared_aliases_preserved=True, downstream_mutation_visible=True,
        outer_projection_independent=True, arbitrary_values_not_validated=True,
        rejected_contracts=rejected,
        provider_sha256=hashlib.sha256(Path(inspect.getfile(provider)).read_bytes()).hexdigest(),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=[
            'Source-schema port adapter only; no numerical algorithm or empirical transfer claim.',
            'Upstream replay validates values; caller owns units, alignment, availability and downstream mutation.',
            'No copying, casting, field renaming, training, inference or catalog approval.',
            'Composed graph execution is qualified separately; these checks establish the local container contract.'])


if __name__ == '__main__':
    result = validate()
    (ROOT / 'docs/reviews/osic_feature_projection.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(passed=result['passed'], insertion_orders=result['insertion_orders_checked'],
                         rejected=len(result['rejected_contracts']))))
