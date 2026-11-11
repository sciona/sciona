"""Reusable runtime compared with direct pinned model construction."""
import ast
import gc
import hashlib
import json
from pathlib import Path
import tempfile

import validate_openvaccine_models as setup
from sciona.openvaccine_models import create_model, SOURCE_SHA256
np, tf, ROOT = setup.np, setup.tf, setup.ROOT


def main():
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    source=Path('/private/tmp/sciona_openvaccine_source')
    raw=(source/'scripts/nullrecurrent_inference.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==SOURCE_SHA256
    names={'MSE','get_base','get_ae_model','get_model','forward','res','attention','multi_head_attention','adj_attn','gru_layer','lstm_layer','wave_block','get_optimizer'}
    definitions=[n for n in ast.parse(raw).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    rng=np.random.default_rng(639)
    nodes=rng.uniform(0,1,(2,10,55)).astype(np.float32)
    adj=np.tile(np.eye(10,dtype=np.float32)[None,:,:,None],(2,1,1,8))
    ns=dict(np=np,tf=tf,L=tf.keras.layers,K=tf.keras.backend,losses=tf.keras.losses,LOSS_WGTS=[.3,.3,.3,.05,.05],X_node=nodes,As=adj)
    exec(compile(ast.Module(body=definitions,type_ignores=[]),'<direct-source>','exec'),ns)
    results=[]
    for family in ['autoencoder','gru','lstm','forward','wave']:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(146)
        base=ns['get_base']({},nodes,adj)
        direct=ns['get_ae_model'](base,{}) if family=='autoencoder' else ns['get_model'](base,{},family,nodes,adj)
        expected=direct([nodes,adj],training=False).numpy()
        count=direct.count_params()
        del direct,base
        gc.collect()
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(146)
        runtime=create_model(source,family)
        actual=runtime.model([nodes,adj],training=False).numpy()
        np.testing.assert_array_equal(actual,expected)
        assert runtime.model.count_params()==count
        if family=='autoencoder':
            loss=runtime.train_step(nodes,adj)
        else:
            truth=rng.normal(size=(2,8,5)).astype(np.float32)
            truth[:,::3,:]=np.nan
            weights=np.ones(2,dtype=np.float32)
            before=[v.numpy() for v in runtime.model.weights]
            rejected=0
            bad=[(nodes.astype(np.float64),adj,truth,weights),
                 (nodes,adj[:,:,:,:7],truth,weights),
                 (nodes,adj,np.full_like(truth,np.nan),weights),
                 (nodes,adj,truth,np.zeros_like(weights)),
                 (nodes,adj,truth,None)]
            for args in bad:
                try: runtime.train_step(*args)
                except ValueError: rejected+=1
                else: raise AssertionError('Invalid training input accepted')
            for value,target in zip(runtime.model.weights,before):np.testing.assert_array_equal(value.numpy(),target)
            assert int(runtime.model.optimizer.iterations.numpy())==0
            loss=runtime.train_step(nodes,adj,truth,weights)
            assert runtime.predict(nodes,adj).shape==(2,8,5)
        assert np.isfinite(loss) and int(runtime.model.optimizer.iterations.numpy())==1
        results.append(dict(family=family,direct_source_output_exact=True,parameters=count,optimizer_steps=1))
        print(json.dumps(results[-1]),flush=True)
        del runtime
        gc.collect()
    with tempfile.TemporaryDirectory(prefix='sciona_openvaccine_bad_source_') as temp:
        p=Path(temp);(p/'scripts').mkdir();(p/'scripts/nullrecurrent_inference.py').write_bytes(raw+b'\n')
        try:create_model(p,'gru')
        except ValueError:pass
        else:raise AssertionError('Modified source accepted')
    report=dict(status='passed',synthetic_only=True,families=results,invalid_training_cases=20,
                invalid_inputs_leave_model_unchanged=True,modified_source_rejected=True,
                runtime_sha256=hashlib.sha256((ROOT/'sciona/openvaccine_models.py').read_bytes()).hexdigest(),
                validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Reusable source model factory, input boundaries and one training step per full topology; complete lifecycle still pending.')
    (ROOT/'docs/reviews/competition_openvaccine_runtime.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
