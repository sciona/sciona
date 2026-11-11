"""Run isolated source schedule with a declared NumPy cosine adapter."""
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from sciona.cassava_efficientnet_schedule import learning_rate
from scripts.validate_cassava_source_components import source_helpers


def main():
    source = Path('/private/tmp/sciona_cassava_winner_source/efficientnet.py')
    pins = json.loads(Path('docs/reviews/competition_cassava_winner_source_pins.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == pins['notebooks']['efficientnet']['code_sha256']
    ns = source_helpers(source, {'lrfn'}, {'math': math, 'BATCH_SIZE': 256, 'EPOCHS': 20,
        'tf': SimpleNamespace(math=SimpleNamespace(cos=lambda x: SimpleNamespace(numpy=lambda: np.cos(np.float32(x)))) )})
    expected = [float(ns['lrfn'](epoch)) for epoch in range(20)]
    actual = [learning_rate(epoch) for epoch in range(20)]
    np.testing.assert_allclose(actual, expected, atol=3e-11, rtol=1e-6)
    report = {'approved': False, 'passed': True, 'epoch_comparisons': 20,
              'refit_uses_first_14_values': True, 'tensorflow_executed': False,
              'scope': 'Source schedule with explicit NumPy float32 cosine adapter; runtime parity remains to verify.',
              'maximum_absolute_error': float(np.max(np.abs(np.array(actual) - expected)))}
    paths = ['sciona/cassava_efficientnet_schedule.py', 'tests/test_cassava_efficientnet_schedule.py',
             'scripts/validate_cassava_efficientnet_schedule.py',
             'scripts/validate_cassava_source_components.py', 'docs/reviews/competition_cassava_winner_source_pins.json']
    report['sha256'] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    Path('docs/reviews/competition_cassava_efficientnet_schedule_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
