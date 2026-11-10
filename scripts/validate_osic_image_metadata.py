"""Synthetic source replay of branch metadata and longitudinal target fitting."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from sciona.affine_trajectory_fit import fit
from sciona.osic_image_metadata import encode,prepare_training
from scripts.review_osic_original_scope import TRAIN,INFERENCE,REPOSITORY,REVISION

ROOT=Path(__file__).resolve().parents[1]
PINS={(TRAIN,15):'1147739a9bf7c186b2676e24e6c5973417b8411a273ab096c60aa24e1319eabe',
      (TRAIN,16):'c01d01b9b003bd9a48868727e1cca308889bf8134c879e245c43bbd3e58b6a14',
      (INFERENCE,6):'56eee99ee8632632a72876cf33e6234c7becfc0f2f8ce3d7e86c8f0a9987a3e7'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(directory):
    code={}
    for (name,index),digest in PINS.items():
        notebook=json.loads((Path(directory)/name).read_text())
        tree=ast.parse(''.join(notebook['cells'][index]['source']))
        assert hashlib.sha256(ast.unparse(tree).encode()).hexdigest()==digest
        code[name,index]=compile(tree,'<pinned-source-preparation>','exec')
    count=0
    for branch,name,index in [('training',TRAIN,15),('inference',INFERENCE,6)]:
        scope={'np':np}
        exec(code[name,index],scope)
        for dtype in (np.int64,np.float32,np.float64):
            for category in ('male','Male','MALE','female','unknown'):
                for status in ('Never smoked','Ex-smoker','Currently smokes','other','never smoked'):
                    value=dtype(47)
                    frame=pd.DataFrame(dict(Age=np.array([value],dtype=dtype),Sex=[category],SmokingStatus=[status]))
                    expected=scope['get_tab'](frame)
                    actual=encode(value,category,status,branch=branch)
                    np.testing.assert_array_equal(actual,expected)
                    assert actual.dtype==expected.dtype
                    count+=1
    assert encode(47,'Male','other',branch='training')[1]==0
    assert encode(47,'Male','other',branch='inference')[1]==1
    table=pd.DataFrame(dict(entity=['b','a','b','c','a','b'],time=[3.,0.,-1.,2.,4.,1.],
        target=[8.,-1.,0.,6.,11.,4.],numeric=[47.,53.,99.,36.,18.,100.],
        category_a=['Male','female','unknown','male','Male','female'],
        category_b=['other','Ex-smoker','Currently smokes','Never smoked','other','other']))
    original=table.copy(deep=True)
    # Only the numerical default is mapped: explicit -1 reproduces the
    # documented pre-2.0 rcond convention in the current NumPy/LAPACK runtime.
    class HistoricalNumpy:
        linalg=SimpleNamespace(lstsq=lambda a,b:np.linalg.lstsq(a,b,rcond=-1))
        def __getattr__(self,name):
            return getattr(np,name)
    scope=dict(np=HistoricalNumpy(),tqdm=lambda it:it,
        train=table.rename(columns=dict(entity='Patient',time='Weeks',target='FVC',
                                       numeric='Age',category_a='Sex',category_b='SmokingStatus')))
    exec(code[TRAIN,15],scope)
    exec(code[TRAIN,16],scope)
    result=prepare_training(table,time_coordinate='synthetic_time',response_coordinate='synthetic_response')
    assert result['entities']==scope['P']==['b','a','c']
    np.testing.assert_array_equal(result['slopes'],[scope['A'][k] for k in scope['P']])
    np.testing.assert_array_equal(result['metadata'],[scope['TAB'][k] for k in scope['P']])
    pd.testing.assert_frame_equal(table,original)
    # Independent affine relations in distinct synthetic interpretations.
    reuse=[]
    for application,coordinate,slope,intercept in [
        ('synthetic_constant_velocity',np.array([-3.,0.,2.,8.]),2.5,-4.),
        ('synthetic_sensor_calibration',np.array([1.,2.,5.,9.]),-.75,12.)]:
        fitted=fit(coordinate,slope*coordinate+intercept,rcond=-1)
        assert fitted['rank']==2
        np.testing.assert_allclose(fitted['coefficients'],[slope,intercept],rtol=1e-13,atol=1e-13)
        reuse.append(dict(application=application,analytic_coefficients_passed=True))
    repeated=fit(np.array([2.,2.,2.]),np.array([4.,6.,8.]),rcond=-1)
    assert repeated['rank']==1
    np.testing.assert_allclose(repeated['coefficients'],[2.4,1.2],rtol=1e-13,atol=1e-13)
    boundary=1.+np.arange(8)*4e-16
    assert fit(boundary,np.arange(8.),rcond=-1)['rank']==2
    assert fit(boundary,np.arange(8.),rcond=None)['rank']==1
    rejected=0
    for x,y,rcond in [([],[],-1),([0.,1.],[1.],-1),([0.,np.nan],[1.,2.],-1),
                     ([True],[1.],-1),([0.],[1.],True),([0.],[1.],-2),([0.],[1.],'invalid')]:
        try:
            fit(x,y,rcond=rcond)
        except ValueError:
            rejected+=1
        else:
            raise AssertionError('Invalid fitting contract accepted')
    rejected_metadata=0
    for value,category,status,branch in [(np.nan,'male','other','training'),
            ([47.],'male','other','training'),(True,'male','other','training'),
            (47.,None,'other','training'),(47.,'male',None,'training'),
            (47.,'male','other','unknown')]:
        try:
            encode(value,category,status,branch=branch)
        except ValueError:
            rejected_metadata+=1
        else:
            raise AssertionError('Invalid metadata contract accepted')
    return dict(format='osic-image-metadata.v1',passed=True,approved=False,catalog_mutations=0,
        synthetic_only=True,source=dict(repository=REPOSITORY,revision=REVISION,
            normalized_cell_sha256=[dict(software_path=n,cell=i,sha256=h) for (n,i),h in PINS.items()]),
        exact_metadata_cases=count,branch_case_asymmetry_preserved=True,
        synthetic_longitudinal_entities=3,first_encounter_order_and_first_metadata_preserved=True,
        single_observation_minimum_norm_preserved=True,repeated_coordinate_minimum_norm_passed=True,
        rank_cutoff_boundary=dict(historical_rank=2,current_default_rank=1),
        invalid_metadata_contracts_rejected=rejected_metadata,
        invalid_fit_contracts_rejected=rejected,constructed_reuse_examples=reuse,
        rank_policy=dict(rcond=-1,numpy=np.__version__,
            historical_documentation='https://numpy.org/doc/1.19/reference/generated/numpy.linalg.lstsq.html',
            current_documentation='https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html'),
        fit_sha256=sha(ROOT/'sciona/affine_trajectory_fit.py'),
        provider_sha256=sha(ROOT.parent/'sciona-atoms/src/sciona/atoms/numerical/statistics/affine_fit.py'),
        adapter_sha256=sha(ROOT/'sciona/osic_image_metadata.py'),validator_sha256=sha(__file__),
        limitations=['Historical NumPy default convention is reconstructed explicitly; historical binary/LAPACK equivalence is unqualified.',
            'No clinical validity, observation availability, raw image decoding or source exclusion-list qualification.',
            'Unknown categories retain source fallbacks; this is not a generic category interpretation.',
            'Rank-deficient slopes are minimum-norm outputs and are not uniquely identified trends.',
            'The ongoing image training run uses separately constructed metadata and targets; this preparation is not yet connected to that run.',
            'No original workflow or catalog approval.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory',required=True)
    report=validate(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/osic_image_metadata.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,metadata_cases=report['exact_metadata_cases'],
                         invalid_fit_contracts_rejected=report['invalid_fit_contracts_rejected'])))
