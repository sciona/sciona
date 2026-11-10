"""Independent synthetic qualification of the reusable uncentered affine fit."""
import hashlib
import json
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np

from sciona.atoms.numerical.statistics.affine_fit import affine_least_squares

ROOT=Path(__file__).resolve().parents[1]


def validate():
    rng=np.random.default_rng(728)
    cases=[]
    with localcontext() as context:
        context.prec=60
        for application in ('synthetic_motion','synthetic_sensor_calibration'):
            for i in range(64):
                x=rng.integers(-100,100,size=17).astype(np.float64)/8
                slope,intercept=(2.5,-4.) if application=='synthetic_motion' else (-.75,12.)
                y=slope*x+intercept+rng.normal(0,.1,size=len(x))
                before=(x.copy(),y.copy())
                xd=list(map(Decimal.from_float,x));yd=list(map(Decimal.from_float,y));n=Decimal(len(x))
                sx,sy=sum(xd),sum(yd)
                a=(n*sum(u*v for u,v in zip(xd,yd))-sx*sy)/(n*sum(u*u for u in xd)-sx*sx)
                b=(sy-a*sx)/n
                residual=sum((v-a*u-b)**2 for u,v in zip(xd,yd))
                got=affine_least_squares(x,y,-1)
                assert got['rank']==2
                np.testing.assert_allclose(got['coefficients'],[float(a),float(b)],rtol=1e-12,atol=1e-12)
                np.testing.assert_allclose(got['residual_sums_of_squares'],[float(residual)],rtol=1e-11,atol=1e-12)
                np.testing.assert_array_equal(x,before[0]);np.testing.assert_array_equal(y,before[1])
            cases.append(dict(application=application,constructed_cases=64,independent_decimal_precision=60,
                coefficients_and_residuals_verified=True,inputs_unchanged=True))
    single=affine_least_squares(np.array([2.]),np.array([6.]),-1)
    assert single['rank']==1 and single['residual_sums_of_squares'].size==0
    np.testing.assert_allclose(single['coefficients'],[2.4,1.2],rtol=1e-13,atol=1e-13)
    rejected=0
    for x,y,r in [([],[],-1),([0.,1.],[1.],-1),([np.nan],[1.],-1),([True],[1.],-1),
                  ([0.],[1.],True),([0.],[1.],-2),([0.],[1.],1),([0.],[1.],np.inf)]:
        try:
            affine_least_squares(np.asarray(x),np.asarray(y),r)
        except ValueError:
            rejected+=1
        else:
            raise AssertionError('Invalid fit accepted')
    provider=ROOT.parent/'sciona-atoms/src/sciona/atoms/numerical/statistics/affine_fit.py'
    return dict(format='affine-provider-qualification.v1',passed=True,approved=False,catalog_mutations=0,
        synthetic_only=True,cases=cases,minimum_norm_single_observation=True,invalid_cases_rejected=rejected,
        provider_sha256=hashlib.sha256(provider.read_bytes()).hexdigest(),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),numpy=np.__version__,
        limitations=['Concrete in-process qualification only; abstract execution intentionally rejects.',
            'Constructed applications establish numerical reuse, not empirical predictive performance.',
            'Registration metadata, transactional publication and served catalog execution remain pending.'])


if __name__=='__main__':
    result=validate()
    (ROOT/'docs/reviews/affine_provider_qualification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,cases=128,invalid_cases_rejected=result['invalid_cases_rejected'])))
