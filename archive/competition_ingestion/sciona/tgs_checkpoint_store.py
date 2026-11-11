"""Local tensor-only, digest-verified TGS phase handoff storage.

The caller chooses a private runtime directory. Checkpoints and receipts stay
outside source control; these are model weights, not resumable optimizer state.
"""
import hashlib
import io
import os
from pathlib import Path
import re
import tempfile

import torch


class CheckpointStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _state(state):
        if not isinstance(state, dict) or not state:
            raise ValueError('nonempty tensor state required')
        for key, value in state.items():
            if (not isinstance(key, str) or not isinstance(value, torch.Tensor)
                    or value.layout != torch.strided or value.is_quantized
                    or not torch.isfinite(value).all()):
                raise ValueError('finite dense named tensors required')

    def put(self, state):
        self._state(state)
        stream = io.BytesIO()
        torch.save({k: v.detach().cpu().clone() for k, v in state.items()}, stream)
        data = stream.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        target = self.directory / (digest + '.pt')
        fd, temporary = tempfile.mkstemp(prefix='.checkpoint-', dir=self.directory)
        try:
            with os.fdopen(fd, 'wb') as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return dict(sha256=digest, bytes=len(data), tensor_count=len(state))

    def get(self, receipt):
        if (not isinstance(receipt, dict) or not isinstance(receipt.get('sha256'), str)
                or not re.fullmatch(r'[0-9a-f]{64}', receipt['sha256'])
                or type(receipt.get('bytes')) is not int or receipt['bytes'] < 1
                or type(receipt.get('tensor_count')) is not int or receipt['tensor_count'] < 1):
            raise ValueError('valid checkpoint receipt required')
        data = (self.directory / (receipt['sha256'] + '.pt')).read_bytes()
        if len(data) != receipt['bytes'] or hashlib.sha256(data).hexdigest() != receipt['sha256']:
            raise ValueError('checkpoint digest or length mismatch')
        state = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
        self._state(state)
        if len(state) != receipt['tensor_count']:
            raise ValueError('checkpoint tensor count mismatch')
        return state


def persist_phase_result(store, result, *, branch):
    """Persist exact selected phase weights, retaining only receipts in result.

    Source phase handoffs intentionally reset optimizers. This does not claim
    interruption recovery inside a fit or infer any checkpoint from epoch count.
    """
    if branch == 'keras':
        fields = ('best_state', 'periodic_states')
    elif branch == 'pytorch':
        fields = ('cycle_states',)
    else:
        raise ValueError('unknown training branch')
    if any(field not in result for field in fields):
        raise ValueError('missing selected phase weights')
    output = {k: v for k, v in result.items() if k not in fields}
    if branch == 'keras':
        output['best_checkpoint'] = store.put(result['best_state'])
        output['periodic_checkpoints'] = {epoch: store.put(state) for epoch, state in result['periodic_states'].items()}
    else:
        output['cycle_checkpoints'] = {epoch: store.put(state) for epoch, state in result['cycle_states'].items()}
    return output
