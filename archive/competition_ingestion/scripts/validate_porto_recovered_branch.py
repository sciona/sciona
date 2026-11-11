"""Run recovered DAE/classifier dimensions and epochs through the joined branch."""
import hashlib
import json
from pathlib import Path
import numpy as np
from sciona.porto_neural_branch import fit_branch
from sciona.porto_transforms import rank_gauss_population
ROOT=Path(__file__).resolve().parents[1]


def main():
    if not __debug__:raise RuntimeError('Assertions required')
    rng=np.random.default_rng(5);z=rng.uniform(-1,1,(128,1));axis=np.linspace(-.95,.95,8)[:,None]
    raw=np.column_stack((z,z*.7,-z*.4,z*.2));query=np.column_stack((axis,axis*.7,-axis*.4,axis*.2))
    values=rank_gauss_population(np.vstack((raw,query)),binary_columns=[])
    dae=dict(hidden=[1500,1500,1500],feature_layers=[0,1,2],epochs=1000,batch_size=128,
        learning_rate=.003,decay=.995,swap_probability=.07,momentum=0.)
    neural=dict(hidden=[1000,1000],epochs=200,batch_size=128,learning_rate=.0001,decay=.995,
        l2=.05,momentum=0.,dropout=.5,input_dropout=.1,dropout_scaling='inverted')
    paths=[ROOT/'sciona'/p for p in ['porto_neural_branch.py','porto_autoencoder.py','porto_neural.py','porto_transforms.py']]+[Path(__file__).resolve()]
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();before={str(p.relative_to(ROOT)):sha(p) for p in paths}
    print('Starting full recovered-width DAE (1000 epochs) and classifier (200 epochs)',flush=True)
    result=fit_branch(values[:128],(z[:,0]>0).astype(int),values[128:],seed=7,dae_controls=dae,neural_controls=neural)
    assert result['learned_width']==4500 and result['dae_epochs']==1000 and len(result['training_loss'])==200
    assert result['dae_final_clean_mse']<result['dae_initial_clean_mse']*.8
    scores=result['probabilities'];assert scores.shape==(8,) and np.isfinite(scores).all() and ((scores>=0)&(scores<=1)).all()
    assert before=={str(p.relative_to(ROOT)):sha(p) for p in paths}
    report=dict(status='passed',approved=False,catalog_mutations=0,synthetic_only=True,
        dae_controls=dae,neural_controls=neural,learned_features=4500,dae_population_rows=136,
        initial_dae_mse=result['dae_initial_clean_mse'],final_dae_mse=result['dae_final_clean_mse'],
        initial_classifier_training_loss=result['training_loss'][0],final_classifier_training_loss=result['training_loss'][-1],
        sha256=before,limits=['One recovered configuration with independent SGD, dropout scaling and all-hidden extraction.',
        'Dimensions/epochs and actual feature-only integration verified; classifier accuracy is not a pass criterion or a historical performance claim.',
        'Four synthetic inputs, no historical training volume/GPU qualification; remaining ensemble configurations unresolved.'])
    (ROOT/'docs/reviews/competition_porto_recovered_branch_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='passed',learned_features=4500,dae_epochs=1000,classifier_epochs=200,
        initial_classifier_loss=result['training_loss'][0],final_classifier_loss=result['training_loss'][-1])),flush=True)


if __name__=='__main__':main()
