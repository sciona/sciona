"""Full twenty-member trained teacher/student refinement on synthetic inputs."""
import gc
import hashlib
import json
from pathlib import Path
import tempfile

import validate_openvaccine_models as setup
from sciona.openvaccine_models import create_model
from sciona.openvaccine_checkpoints import save_checkpoint
from sciona.openvaccine_teacher import MemberCheckpoint,predict_teacher
from sciona.openvaccine_round import refine_round
np,tf,ROOT=setup.np,setup.tf,setup.ROOT


def main():
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    source='/private/tmp/sciona_openvaccine_source'
    rng=np.random.default_rng(751)
    def features():return rng.uniform(0,1,(2,10,55)).astype(np.float32),np.tile(np.eye(10,dtype=np.float32)[None,:,:,None],(2,1,1,8))
    n,a=features();pn,pa=features();vn,va=features()
    t=rng.normal(size=(2,8,5)).astype(np.float32);vt=rng.normal(size=t.shape).astype(np.float32);w=np.ones(2,dtype=np.float32)
    tf.keras.utils.set_random_seed(743)
    ae=create_model(source,'autoencoder');ae.train_step(n,a);base=ae.base.get_weights();del ae;gc.collect()
    with tempfile.TemporaryDirectory(prefix='sciona_openvaccine_synthetic_round_') as temp:
        root=Path(temp);records=[];recipes={}
        for family in ['lstm','gru','forward','wave']:
            for slot in range(5):
                tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(341+slot)
                runtime=create_model(source,family,base_weights=base);runtime.train_step(n,a,t,w)
                directory=root/'initial'/(family+str(slot));digest=save_checkpoint(runtime,directory)
                records.append(MemberCheckpoint(family,slot,directory,digest))
                eligible=np.ones(t.shape,dtype=bool);eligible[:,0,:]=False
                recipes[(family,slot)]=dict(supervised_batches=[(n,a,t,w)],validation_batch=(vn,va,vt,w),
                    eligible=eligible,maximum_uncertainty=100.,perturbations=np.zeros_like(t),sample_weights=w,
                    reverse_flags=np.array([slot%2==0,slot%2!=0]))
                del runtime;gc.collect()
            print('Initial trained family completed: '+family,flush=True)
        refined,history=refine_round(source,records,pn,pa,recipes=recipes,output_directory=root/'refined',
            std_ddof=0,rollback_policy='model_and_optimizer',absolute_tolerance=0.)
        assert len(refined)==len(history)==20
        assert all(h['supervised_steps']==1 and h['pseudo_label_section']['steps']==1 for h in history)
        assert [h['parent_manifest_sha256'] for h in history]==[r.manifest_sha256 for r in records]
        assert all(hashlib.sha256((r.directory/'manifest.json').read_bytes()).hexdigest()==r.manifest_sha256 for r in records)
        print('All twenty student updates and checkpoints completed',flush=True)
        prediction=predict_teacher(source,refined,vn,va,std_ddof=0)
        assert all(x.shape==t.shape and np.isfinite(x).all() for x in prediction.values())
        assert all(h['pseudo_label_section']['optimizer_iteration']==(2 if h['pseudo_label_section']['rolled_back'] else 3) for h in history)
        rollback_count=sum(h['pseudo_label_section']['rolled_back'] for h in history)
        second,second_history=refine_round(source,refined,pn,pa,recipes=recipes,output_directory=root/'second',
            std_ddof=0,rollback_policy='model_and_optimizer',absolute_tolerance=0.)
        assert len(second)==len(second_history)==20
        assert [h['parent_manifest_sha256'] for h in second_history]==[r.manifest_sha256 for r in refined]
        assert all(hashlib.sha256((r.directory/'manifest.json').read_bytes()).hexdigest()==r.manifest_sha256 for r in records+list(refined))
        for first,later in zip(history,second_history):
            previous=first['pseudo_label_section']['optimizer_iteration']
            attempted=later['pseudo_label_section']
            assert attempted['optimizer_iteration']==previous+(1 if attempted['rolled_back'] else 2)
        print('Second complete student generation saved',flush=True)
        final=predict_teacher(source,second,vn,va,std_ddof=0)
        assert all(x.shape==t.shape and np.isfinite(x).all() for x in final.values())
        second_rollback_count=sum(h['pseudo_label_section']['rolled_back'] for h in second_history)
    result=dict(status='passed',synthetic_only=True,trained_teacher_members=20,refined_student_members=40,refinement_rounds=2,
                autoencoder_steps=1,initial_supervised_steps=20,refinement_supervised_steps=40,pseudo_label_steps_attempted=40,
                actual_validation_rollbacks=[rollback_count,second_rollback_count],parent_manifests_unchanged=True,
                refined_checkpoint_ensemble_prediction='passed',
                scope='Two consecutive full reconstructed teacher/student rounds on synthetic supplied populations and explicit recipe; historical folds, folding engine and complete graph remain unverified.',
                runtime_sha256=hashlib.sha256((ROOT/'sciona/openvaccine_round.py').read_bytes()).hexdigest(),
                contract_sha256=hashlib.sha256((ROOT/'sciona/openvaccine_round_contract.py').read_bytes()).hexdigest(),
                validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'docs/reviews/competition_openvaccine_round.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
