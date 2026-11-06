"""Guard full training/inference extraction against the committed implementation."""
import ast
from pathlib import Path
import subprocess
from unittest.mock import patch
import numpy as np
from sciona import m5u_pipeline as pipeline
from sciona.m5u_harmonization import restore

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = '59875e1a691606f987c1d353cd5445352ff98cdb'


def test_training_and_inference_body_is_unchanged():
    previous = ast.parse(subprocess.check_output(
        ['git','show',REFERENCE+':sciona/m5u_pipeline.py'],cwd=ROOT,text=True))
    current = ast.parse((ROOT/'sciona/m5u_pipeline.py').read_text())
    old = next(n for n in previous.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    new = next(n for n in current.body if isinstance(n,ast.FunctionDef) and n.name=='run_unrestored')
    assert ast.dump(old.args)==ast.dump(new.args)
    # Only the restoration call and final return move outside the extracted stage.
    assert [ast.dump(n) for n in old.body[:-2]]==[ast.dump(n) for n in new.body[:-1]]
    assert (ROOT/'sciona/m5u_harmonization.py').read_bytes()==subprocess.check_output(
        ['git','show',REFERENCE+':sciona/m5u_harmonization.py'],cwd=ROOT)


def test_reference_wrapper_forwards_all_controls_and_preserves_restoration():
    values=np.arange(1.,85.).reshape(3,14,2)
    before=values.copy()
    raw=dict(unadjusted=values,levels=np.array([1,2]),factors=np.array([.5,2.]),
        adjustment=.7,quantiles=(.1,.5,.9),hierarchy_indices=np.array([0,1]),
        reports={1:{'models':9}},normalization='retained_base_reference',scope='synthetic')
    progress=object();arguments=[object() for _ in range(5)]
    with patch.object(pipeline,'run_unrestored',return_value=raw) as stage:
        actual=pipeline.run(*arguments,minimum_day=302,seed=718,speed=False,super_speed=False,
                            capacity=3_000_000,progress=progress)
    stage.assert_called_once_with(*arguments,minimum_day=302,seed=718,speed=False,super_speed=False,
                                  capacity=3_000_000,progress=progress)
    expected=restore(values,raw['levels'],raw['factors'],raw['quantiles'],adjustment=.7)
    assert actual['predictions'].tobytes()==expected.tobytes()
    assert values.tobytes()==before.tobytes()
    assert set(actual)=={'predictions','levels','quantiles','hierarchy_indices','reports','normalization','scope'}
    for key in set(actual)-{'predictions'}:
        assert actual[key] is raw[key]
