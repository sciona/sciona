"""Exercise the recovered Porto DAE width/epochs on synthetic inputs only."""
import hashlib
import json
from pathlib import Path
import numpy as np
from sciona.porto_autoencoder import fit_population
from sciona.porto_transforms import rank_gauss_population
ROOT=Path(__file__).resolve().parents[1]


def main():
    if not __debug__:raise RuntimeError('Assertions required')
    rng=np.random.default_rng(5);z=rng.uniform(-1,1,(128,1))
    x=rank_gauss_population(np.column_stack((z,z*.7,z*-.4,z*.2)),binary_columns=[])
    controls=dict(hidden=[1500,1500,1500],feature_layers=[0,1,2],epochs=1000,batch_size=128,
        learning_rate=.003,decay=.995,swap_probability=.07,momentum=0.)
    paths=[ROOT/'sciona/porto_autoencoder.py',ROOT/'sciona/porto_transforms.py',Path(__file__).resolve()]
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    before={str(p.relative_to(ROOT)):sha(p) for p in paths}
    print('Starting recovered three-by-1500 DAE, 1000 epochs, full 128-row batches',flush=True)
    fitted=fit_population(x,seed=7,controls=controls)
    assert len(fitted.clean_mse)==1001 and len(fitted.noisy_training_mse)==1000
    assert fitted.clean_mse[-1]<fitted.clean_mse[0]*.8
    predictions=fitted.reconstruct(x)
    np.testing.assert_allclose(np.mean((predictions-x.astype(np.float32))**2),fitted.clean_mse[-1],rtol=1e-6)
    features=fitted.transform(x[:7])
    assert features.shape==(7,4500) and np.isfinite(features).all()
    assert before=={str(p.relative_to(ROOT)):sha(p) for p in paths}
    report=dict(status='passed',approved=False,catalog_mutations=0,synthetic_only=True,controls=controls,
        initial_clean_mse=fitted.clean_mse[0],final_clean_mse=fitted.clean_mse[-1],epochs=1000,
        synthetic_rows=128,synthetic_input_features=4,extracted_features=4500,sha256=before,
        limits=['Recovered width, epochs, rate, decay, noise and batch size exercised with independent zero-momentum SGD and PyTorch initialization.',
                'All-hidden-layer extraction is an explicit independent choice for this configuration.',
                'Four synthetic input features; historical input width, training population, optimizer/CUDA equivalence and final six-model ensemble unqualified.'])
    (ROOT/'docs/reviews/competition_porto_recovered_dae_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='passed',epochs=1000,initial_clean_mse=fitted.clean_mse[0],final_clean_mse=fitted.clean_mse[-1],extracted_features=4500)),flush=True)


if __name__=='__main__':main()
