"""All twenty full model members and independent stacked ensemble reference."""
import gc
import hashlib
import json
from pathlib import Path

import validate_openvaccine_models as setup
from sciona.openvaccine_models import create_model
from sciona.openvaccine_ensemble import predict_twenty
np,tf,ROOT=setup.np,setup.tf,setup.ROOT


def main():
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    rng=np.random.default_rng(36)
    nodes=rng.uniform(0,1,(1,10,55)).astype(np.float32)
    adj=np.tile(np.eye(10,dtype=np.float32)[None,:,:,None],(1,1,1,8))
    saved_nodes,saved_adj=nodes.copy(),adj.copy()
    expected=[]
    def members():
        for family in ['lstm','gru','forward','wave']:
            for slot in range(5):
                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(201+slot)
                runtime=create_model('/private/tmp/sciona_openvaccine_source',family)
                a=runtime.model([nodes,adj],training=False).numpy()
                b=runtime.model([nodes[:,::-1,:].copy(),adj[:,::-1,::-1,:].copy()],training=False).numpy()
                expected.append((a+b[:,::-1,:])/2)
                yield family,slot,runtime
                del runtime
                gc.collect()
            print(f'Completed {family} five members',flush=True)
    result=predict_twenty(members(),nodes,adj,std_ddof=0)
    stacked=np.array(expected,dtype=np.float64)
    np.testing.assert_allclose(result['prediction'],np.mean(np.clip(stacked,-.5,6),axis=0),rtol=2e-6,atol=1e-7)
    np.testing.assert_allclose(result['teacher_mean'],np.mean(stacked,axis=0),rtol=2e-6,atol=1e-7)
    np.testing.assert_allclose(result['teacher_std'],np.std(stacked,axis=0),rtol=2e-6,atol=1e-7)
    np.testing.assert_array_equal(nodes,saved_nodes);np.testing.assert_array_equal(adj,saved_adj)
    class Fixed:
        def __init__(self,family,value):self.family,self.value=family,value
        def predict(self,n,a):return np.full((1,8,5),self.value,dtype=np.float32)
    fixed=[(f,s,Fixed(f,10 if s%2 else -2)) for f in ['lstm','gru','forward','wave'] for s in range(5)]
    a=predict_twenty(fixed,nodes,adj,std_ddof=1)
    b=predict_twenty(list(reversed(fixed)),nodes,adj,std_ddof=1)
    np.testing.assert_allclose(a['prediction'],2.1,rtol=1e-6)
    for key in a:np.testing.assert_allclose(a[key],b[key],rtol=1e-6,atol=1e-7)
    for bad in [fixed[:-1],fixed+[fixed[0]]]:
        try:predict_twenty(bad,nodes,adj,std_ddof=0)
        except ValueError:pass
        else:raise AssertionError('Incomplete/duplicate ensemble accepted')
    report=dict(status='passed',synthetic_only=True,full_model_members=20,reverse_averaging=True,
                stacked_reference_comparison='passed',teacher_std_ddof=0,
                symmetric_clipping_order_comparison=True,incomplete_duplicate_rejected=True,
                caller_inputs_unchanged=True,
                scope='Full twenty-state topology inference with synthetic initialization; corrected all-member clipping and explicit teacher moments, not trained-state or historical-final-blend equivalence.',
                runtime_sha256=hashlib.sha256((ROOT/'sciona/openvaccine_ensemble.py').read_bytes()).hexdigest(),
                validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'docs/reviews/competition_openvaccine_ensemble.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
