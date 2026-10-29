"""Qualify the unchanged legacy framing primitive for canonical Tier 3 reuse."""
import hashlib
import inspect
import json
from pathlib import Path
import icontract
import numpy as np
from sciona.ghost.abstract import AbstractArray
from sciona.atoms.audio_speech import atoms, witnesses

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = 'be40e926-d0ed-4c8b-83d4-740ef0609355'
VERSION = 'a00a4320-92d1-5c10-aa72-361d5494040f'
CONTENT_HASH = '1deccd1d0c1230068c6f84be92f20e5823ca438a309dd174f086d69527555fec'
FQDN = 'sciona.atoms.audio_speech.audio_windows'
RUNTIME = 'sciona.atoms.audio_speech.atoms.audio_windows'
DESCRIPTION = 'Frame a finite one-dimensional float64 signal into complete windows using positive integer window and hop sizes in samples. Return a read-only float64 view; trailing incomplete windows are dropped. Values and units are preserved, and the view aliases the caller-owned signal.'
BOUND = 'Finite nonempty one-dimensional NumPy float64 array; positive integer window and hop sizes, with at least one complete window. Read-only output aliases the input: later input mutations remain visible. No padding, resampling, time inference or domain-specific interpretation.'
PORTS = [
    dict(direction='input', name='signal', ordinal=0, type_desc='numpy.ndarray', constraints='Finite one-dimensional float64 array covering at least one window; caller owns sample units and ordering.'),
    dict(direction='input', name='window_size', ordinal=1, type_desc='int', constraints='Positive integer window length in samples, no larger than signal length.'),
    dict(direction='input', name='hop_size', ordinal=2, type_desc='int', constraints='Positive integer step in samples; overlap or gaps are explicit.'),
    dict(direction='output', name='windows', ordinal=0, type_desc='numpy.ndarray', constraints='Read-only float64 view with shape ((len(signal)-window_size)//hop_size+1,window_size), preserving input units and aliasing input storage.'),
]
for p in PORTS:
    p.update(required=True, default_value_repr='')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate():
    require(sha(atoms.__file__) == 'b7203d541172eee357c40f968bdef12abff0d3b5255775e1076577d67ceb9462', 'Ensemble-qualified provider changed')
    require(list(inspect.signature(atoms.audio_windows).parameters) == ['signal', 'window_size', 'hop_size'], 'Callable contract changed')
    cases = 0
    samples = np.arange(67, dtype=np.float64)
    # Constructed sensor traces from two distinct application interpretations.
    for signal in [np.sin(samples/7)+samples/100, 100+samples**2/13]:
        for view in [signal, signal[::2], signal[::-1]]:
            for size, hop in [(1,1), (4,2), (5,5), (3,7), (len(view),1), (7,3)]:
                before = view.copy()
                expected = np.array([[view[start+offset] for offset in range(size)]
                                     for start in range(0, len(view)-size+1, hop)])
                result = atoms.audio_windows(view, size, hop)
                np.testing.assert_array_equal(result, expected)
                np.testing.assert_array_equal(view, before)
                require(result.dtype == np.float64 and not result.flags.writeable and np.shares_memory(result, view), 'Read-only alias contract differs')
                abstract = witnesses.witness_audio_windows(AbstractArray(shape=view.shape, dtype='float64'), size, hop)
                require(tuple(abstract.shape) == result.shape and abstract.dtype == 'float64', 'Witness shape differs')
                try:
                    result[0,0] = 0
                except ValueError:
                    pass
                else:
                    raise ValueError('Output accepted a write')
                saved = view[0]
                view[0] = saved+1
                require(result[0,0] == view[0], 'Input alias semantics changed')
                view[0] = saved
                cases += 1
    invalid = [(np.ones((2,4)),2,1), (np.array([]),1,1), (np.array([np.nan,1.]),1,1),
               (np.array([np.inf,1.]),1,1), (np.ones(4),0,1), (np.ones(4),2,0),
               (np.ones(4),5,1), (np.ones(4),-1,1), (np.ones(4),2,-1)]
    for signal, size, hop in invalid:
        try:
            atoms.audio_windows(signal,size,hop)
        except (ValueError, icontract.ViolationError):
            pass
        else:
            raise ValueError('Invalid geometry or signal accepted')
    return dict(passed=True, approved=False, synthetic_only=True, catalog_mutations=0,
                artifact_id=ARTIFACT, version_id=VERSION, content_hash=CONTENT_HASH,
                runtime_fqdn=RUNTIME, numerical_cases=cases, invalid_cases=len(invalid),
                constructed_applications=['vibration sensor framing', 'energy measurement framing'],
                independent_index_oracle=True, noncontiguous_inputs=True, readonly_alias_verified=True,
                witness_verified=True, numpy_version=np.__version__, icontract_version=icontract.__version__,
                provider_sha256=sha(atoms.__file__), witness_sha256=sha(witnesses.__file__),
                validator_sha256=sha(__file__), description=DESCRIPTION, validity_bound=BOUND, ports=PORTS,
                limitations=['Constructed signal examples establish framing behavior, not empirical application performance.',
                    'Qualification is for float64 inputs, as required by the ensemble boundary. Other dtypes are not added to the contract.',
                    'Read-only does not mean immutable: caller mutation of the input changes the aliased output.',
                    'No source provider edits, catalog mutation or approval in this validator.'])


if __name__ == '__main__':
    report = validate()
    (ROOT/'docs/reviews/audio_windows_canonical_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['passed','numerical_cases','invalid_cases','catalog_mutations']}))
