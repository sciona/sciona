"""Exercise complete default pseudo-label counts on a synthetic population."""
import hashlib
import json
from pathlib import Path
import numpy as np
from sciona.santander_lifecycle import fit
from sciona.santander_blend import blend_neural_tree
ROOT=Path(__file__).resolve().parents[1]


def main():
    if not __debug__:raise RuntimeError('Assertions required')
    rng=np.random.default_rng(81);y=np.tile([0,1],30)
    x=y[:,None]*2+rng.normal(0,.5,(60,4))
    axis=np.linspace(-1,3,8500)
    q=axis[:,None]*np.array([1.,1.1,.9,.8])[None,:]
    neural=dict(folds=10,seeds=(42,),epochs=15,batch_size=1024)
    tree=dict(folds=10,seeds=(42,),controls=dict(num_leaves=3,learning_rate=.1,feature_fraction=1.,
        bagging_fraction=1.,bagging_freq=0,min_data_in_leaf=2,max_rounds=8,stopping_rounds=2,categorical=True))
    paths=sorted((ROOT/'sciona').glob('santander_*.py'))+[Path(__file__).resolve()]
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    before={str(p.relative_to(ROOT)):sha(p) for p in paths}
    print('Starting default-tail synthetic lifecycle: 30 models, 10 folds, 15 neural epochs',flush=True)
    result=fit(x,y,q,initial_neural_controls=neural,neural_controls=neural,tree_controls=tree,reference_policy='retain_selected')
    assert result['selection_counts']=={
        'neural':dict(positive=5000,negative=3000,labeled_rows=8060,query_reference_rows=8500),
        'tree':dict(positive=2700,negative=2000,labeled_rows=4760,query_reference_rows=8500)}
    assert len(result['initial_neural']['models'])==10
    assert all(len(branch['models'])==10 for branch in result['branches'].values())
    assert result['scores'].shape==(8500,) and np.isfinite(result['scores']).all()
    np.testing.assert_array_equal(result['scores'],blend_neural_tree(result['branches']['neural']['mean_ranks'],result['branches']['tree']['mean_ranks']))
    assert before=={str(p.relative_to(ROOT)):sha(p) for p in paths}
    report=dict(status='passed',approved=False,catalog_mutations=0,synthetic_only=True,models=30,folds_per_stage=10,
        neural_epochs=15,selection_counts=result['selection_counts'],sha256=before,
        limits=['Full documented tail counts on four synthetic feature groups; historical feature width and training volume not reproduced.',
                'Explicit retained selected-query reference and joint pseudo-label folds; historical membership unresolved.',
                'No independent generalization, competitive accuracy or publication claim.'])
    (ROOT/'docs/reviews/competition_santander_default_tail_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='passed',models=30,selection_counts=result['selection_counts'])),flush=True)


if __name__=='__main__':main()
