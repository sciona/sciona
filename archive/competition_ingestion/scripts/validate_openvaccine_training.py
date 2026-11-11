"""All four reconstructed member lifecycles on separate synthetic populations."""
import gc
import hashlib
import json
from pathlib import Path
import tempfile

import validate_openvaccine_models as setup
from sciona.openvaccine_training import train_member
from sciona.openvaccine_checkpoints import load_checkpoint
from sciona.openvaccine_targets import uncertainty_targets
from sciona.openvaccine_population import population_weights,reverse_training_batch
np,tf,ROOT=setup.np,setup.tf,setup.ROOT


def main():
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    rng=np.random.default_rng(533)
    def features():
        return rng.uniform(0,1,(2,10,55)).astype(np.float32),np.tile(np.eye(10,dtype=np.float32)[None,:,:,None],(2,1,1,8))
    n,a=features();pn,pa=features();vn,va=features()
    truth=rng.normal(size=(2,8,5)).astype(np.float32)
    weights=population_weights(np.array([1,1]),np.ones(2,dtype=np.float32))
    supervised=reverse_training_batch(n,a,truth,weights,np.array([True,False]))
    pseudo_mean=rng.normal(size=(2,8,5)).astype(np.float32)
    std=np.full_like(pseudo_mean,.25)
    eligible=np.ones(pseudo_mean.shape,dtype=bool);eligible[:,0,:]=False
    pseudo=uncertainty_targets(pseudo_mean,std,eligible,maximum_uncertainty=.5,perturbations=np.zeros_like(std))
    pl_batch=reverse_training_batch(pn,pa,pseudo,weights,np.array([False,True]))
    validation_targets=rng.normal(size=(2,8,5)).astype(np.float32)
    validation_targets[:,::3,:]=np.nan
    validation=(vn,va,validation_targets,weights)
    summaries=[]
    with tempfile.TemporaryDirectory(prefix='sciona_openvaccine_synthetic_training_') as temp:
        for family in ['gru','lstm','forward','wave']:
            tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(178)
            checkpoint=Path(temp)/family
            runtime,report=train_member('/private/tmp/sciona_openvaccine_source',family,
                pretraining_batches=[(n,a)],supervised_sections=[[supervised],[supervised]],
                pseudo_label_sections=[[pl_batch],[pl_batch]],validation_batch=validation,
                checkpoint_directory=checkpoint,rollback_policy='model_and_optimizer',absolute_tolerance=0.)
            assert [s['kind'] for s in report['sections']]==['supervised','pseudo_label']*2
            assert all(s['steps']==1 for s in report['sections'])
            expected=runtime.predict(vn,va)
            del runtime;gc.collect()
            restored=load_checkpoint('/private/tmp/sciona_openvaccine_source',checkpoint,
                expected_manifest_sha256=report['checkpoint_manifest_sha256'],expected_family=family)
            np.testing.assert_array_equal(restored.predict(vn,va),expected)
            # Do not retain checkpoint digests or private population details in public evidence.
            summary=dict(family=family,pretraining_steps=report['pretraining_steps'],
                         supervised_sections=2,pseudo_label_sections=2,
                         rollback_count=sum(s.get('rolled_back',False) for s in report['sections']),
                         checkpoint_prediction_exact=True)
            summaries.append(summary);print(json.dumps(summary),flush=True)
            del restored;gc.collect()
    result=dict(status='passed',synthetic_only=True,families=summaries,
                separate_synthetic_training_validation_inputs=True,partially_observed_validation=True,
                scope='Four full member pretraining/supervised/PL/checkpoint lifecycles with supplied synthetic teacher statistics; no complete twenty-member teacher generation, fold provenance or historical recipe equivalence.',
                runtime_sha256=hashlib.sha256((ROOT/'sciona/openvaccine_training.py').read_bytes()).hexdigest(),
                validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'docs/reviews/competition_openvaccine_training.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
