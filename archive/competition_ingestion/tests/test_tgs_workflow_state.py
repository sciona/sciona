import numpy as np
import pytest

from sciona.tgs_workflow_state import WorkflowStateStore


def test_roundtrip_preserves_integer_round_keys_masks_and_rng(tmp_path):
    store = WorkflowStateStore(tmp_path, context_sha256='a'*64)
    rng = np.random.default_rng(77)
    state = dict(seed=77, fits={'synthetic': {'rng_state': rng.bit_generator.state}},
                 rounds={1: dict(masks=np.array([[[True, False]]]), area=np.array([1],dtype=np.int64))})
    store.save(state)
    state['rounds'][1]['masks'][:] = False
    loaded = store.load()
    assert loaded['rounds'][1]['masks'].dtype == np.bool_
    assert loaded['rounds'][1]['masks'][0,0,0]
    assert loaded['rounds'][1]['area'].dtype == np.int64
    resumed = np.random.default_rng()
    resumed.bit_generator.state = loaded['fits']['synthetic']['rng_state']
    np.testing.assert_array_equal(rng.random(12), resumed.random(12))


def test_context_change_and_array_corruption_fail(tmp_path):
    store=WorkflowStateStore(tmp_path,context_sha256='a'*64)
    store.save(dict(values=np.arange(3)))
    with pytest.raises(ValueError,match='context differs'):
        WorkflowStateStore(tmp_path,context_sha256='b'*64).load()
    path=next(tmp_path.glob('*.npy'))
    data=bytearray(path.read_bytes());data[-1]^=1;path.write_bytes(data)
    with pytest.raises(ValueError,match='array digest'):
        store.load()


def test_invalid_new_state_leaves_previous_manifest_intact(tmp_path):
    store=WorkflowStateStore(tmp_path,context_sha256='a'*64)
    store.save(dict(completed=1))
    with pytest.raises(ValueError):
        store.save(dict(completed=2,values=np.array([object()],dtype=object)))
    assert store.load()==dict(completed=1)
