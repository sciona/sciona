"""Twenty trained checkpoint members generating independent teacher statistics."""
import gc
import hashlib
import json
from pathlib import Path
import tempfile

import validate_openvaccine_models as setup
from sciona.openvaccine_models import create_model
from sciona.openvaccine_checkpoints import save_checkpoint
from sciona.openvaccine_teacher import MemberCheckpoint,predict_teacher
np,tf,ROOT=setup.np,setup.tf,setup.ROOT


def main():
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    source='/private/tmp/sciona_openvaccine_source'
    rng=np.random.default_rng(762)
    nodes=rng.uniform(0,1,(2,10,55)).astype(np.float32)
    adj=np.tile(np.eye(10,dtype=np.float32)[None,:,:,None],(2,1,1,8))
    unseen=rng.uniform(0,1,(2,10,55)).astype(np.float32)
    targets=rng.normal(size=(2,8,5)).astype(np.float32)
    weights=np.ones(2,dtype=np.float32)
    tf.keras.utils.set_random_seed(248)
    ae=create_model(source,'autoencoder');ae.train_step(nodes,adj)
    learned=ae.base.get_weights()
    del ae;gc.collect()
    predictions=[]
    with tempfile.TemporaryDirectory(prefix='sciona_openvaccine_trained_teacher_') as temp:
        records=[]
        for family in ['lstm','gru','forward','wave']:
            for slot in range(5):
                tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(500+slot)
                runtime=create_model(source,family,base_weights=learned)
                runtime.train_step(nodes,adj,targets,weights)
                a=runtime.predict(unseen,adj)
                b=runtime.predict(unseen[:,::-1,:].copy(),adj[:,::-1,::-1,:].copy())[:,::-1,:]
                predictions.append((a+b)/2)
                directory=Path(temp)/(family+str(slot))
                digest=save_checkpoint(runtime,directory)
                records.append(MemberCheckpoint(family,slot,directory,digest))
                del runtime;gc.collect()
            print(f'Trained and checkpointed five {family} members',flush=True)
        result=predict_teacher(source,records,unseen,adj,std_ddof=0)
        stacked=np.array(predictions,dtype=np.float64)
        for name,expected in [('prediction',np.clip(stacked,-.5,6).mean(axis=0)),
                              ('teacher_mean',stacked.mean(axis=0)),('teacher_std',stacked.std(axis=0))]:
            np.testing.assert_allclose(result[name],expected,rtol=2e-6,atol=1e-7)
        for invalid in [records[:-1],records[:-1]+[records[0]]]:
            try:predict_teacher(source,invalid,unseen,adj,std_ddof=0)
            except ValueError:pass
            else:raise AssertionError('Invalid checkpoint population accepted')
    report=dict(status='passed',synthetic_only=True,autoencoder_pretraining_steps=1,
                trained_checkpoint_members=20,supervised_steps_per_member=1,
                restored_teacher_reference_comparison=True,unseen_synthetic_prediction_inputs=True,
                missing_duplicate_population_rejected=True,
                scope='Complete trained reconstructed teacher from shared synthetic pretrained base; no student refinement rounds, historical fold provenance or historical recipe equivalence.',
                runtime_sha256=hashlib.sha256((ROOT/'sciona/openvaccine_teacher.py').read_bytes()).hexdigest(),
                validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'docs/reviews/competition_openvaccine_teacher.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
