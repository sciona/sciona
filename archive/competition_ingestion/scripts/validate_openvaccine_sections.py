"""Real model updates with actual validation and injected rollback failures."""
import hashlib
import json
from pathlib import Path

import validate_openvaccine_models as setup
from sciona.openvaccine_models import create_model
from sciona.openvaccine_sections import run_pseudo_label_section
np,tf,ROOT=setup.np,setup.tf,setup.ROOT


def exact(left,right):
    assert len(left)==len(right)
    for a,b in zip(left,right):np.testing.assert_array_equal(a,b)


def main():
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(421)
    runtime=create_model('/private/tmp/sciona_openvaccine_source','gru')
    rng=np.random.default_rng(631)
    nodes=rng.uniform(0,1,(2,10,55)).astype(np.float32)
    adj=np.tile(np.eye(10,dtype=np.float32)[None,:,:,None],(2,1,1,8))
    truth=rng.normal(size=(2,8,5)).astype(np.float32)
    weights=np.ones(2,dtype=np.float32)
    batch=(nodes,adj,truth,weights)
    def actual_score(model):
        return float(model.model.loss(tf.constant(truth),tf.constant(model.predict(nodes,adj)),tf.constant(weights)).numpy())
    first=run_pseudo_label_section(runtime,[batch],actual_score,absolute_tolerance=100.,rollback_policy='model_and_optimizer')
    assert not first['rolled_back'] and first['steps']==1
    assert np.isfinite(first['validation_after_attempt'])
    results=[]
    for policy in ['weights_only','model_and_optimizer']:
        before=runtime.model.get_weights()
        opt=[v.numpy() for v in runtime.model.optimizer.variables()]
        iteration=int(runtime.model.optimizer.iterations.numpy())
        values=iter([1.,2.])
        # Synthetic injected metric degradation isolates the rollback branch;
        # training still executes the complete source model with dropout on.
        result=run_pseudo_label_section(runtime,[batch],lambda model: next(values),
                                       absolute_tolerance=.125,rollback_policy=policy)
        assert result['rolled_back']
        exact(runtime.model.get_weights(),before)
        if policy=='model_and_optimizer':exact([v.numpy() for v in runtime.model.optimizer.variables()],opt)
        else:assert int(runtime.model.optimizer.iterations.numpy())==iteration+1
        results.append(result)
    before=runtime.model.get_weights()
    opt=[v.numpy() for v in runtime.model.optimizer.variables()]
    def broken_batches():
        yield batch
        raise RuntimeError('synthetic failure after update')
    try:run_pseudo_label_section(runtime,broken_batches(),actual_score,absolute_tolerance=0.,rollback_policy='weights_only')
    except RuntimeError as error:assert str(error)=='synthetic failure after update'
    else:raise AssertionError('Expected batch failure')
    exact(runtime.model.get_weights(),before)
    exact([v.numpy() for v in runtime.model.optimizer.variables()],opt)
    report=dict(status='passed',synthetic_only=True,actual_validation_section=first,
                injected_metric_rollback_cases=results,post_update_exception_restores_model_optimizer=True,
                source_model='full_gru',stochastic_training_enabled=True,random_stream_rewind=False,
                scope='One real validation section, both injected deterioration policies and failure after actual update; full lifecycle and stochastic replay remain pending.',
                runtime_sha256=hashlib.sha256((ROOT/'sciona/openvaccine_sections.py').read_bytes()).hexdigest(),
                validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'docs/reviews/competition_openvaccine_sections.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
