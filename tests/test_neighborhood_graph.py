"""Registered uncached graph execution against an explicit matrix oracle."""
import numpy as np
import pytest
from sciona.neighborhood_graph import build_neighborhood_graph
from tests.test_text_pair_provider_graphs import run


@pytest.mark.parametrize('strength',[0.,.4,1.])
def test_native_numeric_blending(strength,tmp_path,monkeypatch):
    values=np.array([-6.,3.,9.])
    neighbors=np.array([[False,True,True],[True,False,False],[False,False,False]])
    weights=np.array([[0.,.5,.5],[1.,0.,0.],[0.,0.,1.]])
    captured=run(build_neighborhood_graph(),dict(values=values,neighbors=neighbors,strength=strength),tmp_path,monkeypatch)
    expected=((1-strength)*np.eye(3)+strength*weights)@values
    np.testing.assert_allclose(captured['blend','out_result'],expected)
    np.testing.assert_array_equal(values,[-6.,3.,9.])
