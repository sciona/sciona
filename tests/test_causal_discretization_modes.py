import numpy as np
from sciona.atoms.causal_inference.feature_primitives import atoms as f
from sciona.atoms.causal_inference.conditional_statistics import atoms as c
from sciona.atoms.causal_inference.feature_primitives.batched_statistics import typed_pair_statistic


def test_conditional_similarity_exercises_unnormalized_source_binning(monkeypatch):
    original=f.discretize_and_bin
    observed=[]
    def spy(x,tx,ffactor=3,maxdev=3,norm=True):
        observed.append((ffactor,maxdev,norm))
        result=original(x,tx,ffactor,maxdev,norm)
        if not norm:
            np.testing.assert_array_equal(result,np.clip(np.round(x*2),-6,6))
        return result
    monkeypatch.setattr(f,'discretize_and_bin',spy)
    monkeypatch.setattr(c,'discretize_and_bin',spy)
    rng=np.random.default_rng(425)
    x=np.repeat([0.,1.],40);y=rng.normal(size=80)
    score=typed_pair_statistic([x],['Binary'],[y],['Numerical'],'conditional_similarity')
    assert np.isfinite(score).all() and score[0]>0
    assert (2,3,True) in observed and (2,3,False) in observed
    typed_pair_statistic([y],['Numerical'],[x],['Binary'],'discrete_entropy')
    assert (3,3,True) in observed
