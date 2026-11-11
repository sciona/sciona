"""Synthetic trained state restoration and corruption rejection."""
import hashlib
import json
from pathlib import Path
import tempfile

import validate_openvaccine_models as setup
from sciona.openvaccine_models import create_model
from sciona.openvaccine_checkpoints import save_checkpoint,load_checkpoint
np,tf,ROOT=setup.np,setup.tf,setup.ROOT


def main():
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(287)
    source='/private/tmp/sciona_openvaccine_source'
    runtime=create_model(source,'gru')
    rng=np.random.default_rng(314)
    nodes=rng.uniform(0,1,(2,10,55)).astype(np.float32)
    adj=np.tile(np.eye(10,dtype=np.float32)[None,:,:,None],(2,1,1,8))
    runtime.train_step(nodes,adj,rng.normal(size=(2,8,5)).astype(np.float32),np.ones(2,dtype=np.float32))
    prediction=runtime.predict(nodes,adj)
    rejected=0
    with tempfile.TemporaryDirectory(prefix='sciona_openvaccine_checkpoint_runtime_') as temp:
        directory=Path(temp)/'synthetic'
        digest=save_checkpoint(runtime,directory)
        loaded=load_checkpoint(source,directory,expected_manifest_sha256=digest,expected_family='gru')
        np.testing.assert_array_equal(loaded.predict(nodes,adj),prediction)
        for a,b in zip(loaded.model.weights,runtime.model.weights):np.testing.assert_array_equal(a.numpy(),b.numpy())
        a,b=loaded.model.optimizer.variables(),runtime.model.optimizer.variables()
        assert len(a)==len(b)
        for x,y in zip(a,b):np.testing.assert_array_equal(x.numpy(),y.numpy())
        for expected,family in [('0'*64,'gru'),(digest,'wave')]:
            try:load_checkpoint(source,directory,expected_manifest_sha256=expected,expected_family=family)
            except ValueError:rejected+=1
            else:raise AssertionError('Mismatched identity accepted')
        index=directory/'state-1.index'
        with index.open('ab') as stream:stream.write(b'synthetic corruption')
        try:load_checkpoint(source,directory,expected_manifest_sha256=digest,expected_family='gru')
        except ValueError:rejected+=1
        else:raise AssertionError('Corrupt checkpoint accepted')
        try:save_checkpoint(runtime,directory)
        except FileExistsError:rejected+=1
        else:raise AssertionError('Existing checkpoint overwritten')
    report=dict(status='passed',synthetic_only=True,full_model='gru',trained_optimizer_steps=1,
                prediction_and_model_state_exact=True,optimizer_state_exact=True,negative_cases=rejected,
                scope='Reusable private checkpoint integrity and full model/Adam restoration; no random-stream or complete lifecycle persistence claim.',
                runtime_sha256=hashlib.sha256((ROOT/'sciona/openvaccine_checkpoints.py').read_bytes()).hexdigest(),
                validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'docs/reviews/competition_openvaccine_checkpoint_runtime.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
