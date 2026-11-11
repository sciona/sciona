"""Raw synthetic sequences through folding, explicit splits and full training."""
import hashlib
import json
from pathlib import Path
import warnings
import validate_openvaccine_models as setup
from sciona.openvaccine_ensemble import EXPECTED_MEMBERS
from sciona.openvaccine_lifecycle import execute_population
np,tf,ROOT=setup.np,setup.tf,setup.ROOT


def main():
    tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(582);rng=np.random.default_rng(174)
    def sequences(count,length):return [''.join(rng.choice(list('ACGU'),size=length)) for _ in range(count)]
    labeled=sequences(4,12);pseudo=sequences(2,16);prediction=sequences(2,14)
    targets=rng.normal(size=(4,12,5)).astype(np.float32);targets[:,8:,:]=np.nan
    splits={key:dict(train=[0,1],validation=[2,3]) if key[1]%2==0 else dict(train=[2,3],validation=[0,1]) for key in EXPECTED_MEMBERS}
    plans={key:dict(supervised_reverse_flags=np.array([True,False]),eligible=np.ones((2,16,5),dtype=bool),
                   maximum_uncertainty=100.,perturbations=np.zeros((2,16,5),dtype=np.float32),
                   sample_weights=np.ones(2,dtype=np.float32),reverse_flags=np.array([False,True])) for key in EXPECTED_MEMBERS}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',FutureWarning)
        result=execute_population('/private/tmp/sciona_openvaccine_source',
            binary='/private/tmp/sciona_openvaccine_contrafold/src/contrafold',
            parameters='/private/tmp/sciona_openvaccine_eternafold_parameters/EternaFoldParams.v1',
            sequences=labeled,targets=targets,cluster_ids=np.array([0,0,1,2]),proximity_factors=np.ones(4,dtype=np.float32),
            splits=splits,pseudo_sequences=pseudo,prediction_sequences=prediction,pseudo_rounds=[plans],
            pretraining_steps=1,initial_supervised_steps=1,rollback_policy='model_and_optimizer',absolute_tolerance=0.,std_ddof=0)
    assert result['members']==20 and len(result['rounds'])==1
    assert result['rounds'][0]['members']==result['rounds'][0]['supervised_steps']==result['rounds'][0]['pseudo_label_steps']==20
    assert all(result[k].shape==(2,14,5) and np.isfinite(result[k]).all() for k in ['predictions','teacher_mean','teacher_std'])
    report=dict(status='passed',synthetic_only=True,raw_sequence_folding=True,distinct_population_lengths=True,
                masked_validation=True,explicit_member_splits=20,members=20,rounds=result['rounds'],
                scope='Integrated raw synthetic populations through actual folding, features, pretraining, supervised/PL round and checkpoint ensemble; not serialized graph, historical recipe or independent CV accuracy.',
                runtime_sha256=hashlib.sha256((ROOT/'sciona/openvaccine_lifecycle.py').read_bytes()).hexdigest(),
                contract_sha256=hashlib.sha256((ROOT/'sciona/openvaccine_round_contract.py').read_bytes()).hexdigest(),
                validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'docs/reviews/competition_openvaccine_lifecycle.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
