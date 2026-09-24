"""Execute the stored quadratic graph in synthetic mechanics and cost models."""
import argparse
import asyncio
import json
from pathlib import Path
import tempfile
import numpy as np
import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from scripts.check_quadratic_revision import check
from scripts.stage_quadratic_original_revision import ROOT,sha,require
from sciona.visualizer import runner


def selected(approved):
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
            options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        return check(db,approved)


async def validate(approved=False):
    graph,imported=selected(approved)
    rng=np.random.default_rng(82319);n=128
    early=rng.integers(1,30,n).astype(float);late=early+rng.integers(1,30,n)
    # Constant acceleration: y(t)=v*t-g*t^2/2, g=2 m/s^2.
    # Construct target crossings, then independently check heights and directions.
    gravity=np.full(n,2.);velocity=early+late;target=early*late
    qa=rng.integers(1,30,n).astype(float);qb=qa+rng.integers(1,30,n)
    curvature=rng.choice([0.5,1.,2.],n);variable_cost=rng.integers(1,10,n).astype(float)
    price=variable_cost+curvature*(qa+qb);fixed_cost=curvature*qa*qb
    cases=[('constant_acceleration',dict(a=-gravity/2,b=velocity,c=-target),early,late),
        ('quadratic_revenue_linear_cost',dict(a=-curvature,b=price-variable_cost,c=-fixed_cost),qa,qb)]
    reports=[]
    with tempfile.TemporaryDirectory(prefix='sciona-quadratic-reuse-') as tmp:
        previous=runner.RUNS_DIR;runner.RUNS_DIR=Path(tmp)
        try:
            for name,inputs,low,high in cases:
                result=await runner.CDGExecutionSession(None,'synthetic-reuse',name).execute(inputs,cdg=graph)
                require(result['status']=='completed','Reuse execution failed')
                lo=np.load(Path(tmp)/name/'roots/out_lower_root.npy');hi=np.load(Path(tmp)/name/'roots/out_upper_root.npy')
                np.testing.assert_array_equal(lo,low);np.testing.assert_array_equal(hi,high)
                if name=='constant_acceleration':
                    for roots in [lo,hi]:np.testing.assert_array_equal(velocity*roots-gravity*roots**2/2,target)
                    require(np.all(velocity-gravity*lo>0) and np.all(velocity-gravity*hi<0),'Crossing direction differs')
                else:
                    for roots in [lo,hi]:np.testing.assert_array_equal(price*roots-curvature*roots**2,fixed_cost+variable_cost*roots)
                    middle=(lo+hi)/2
                    require(np.all(price*middle-curvature*middle**2>fixed_cost+variable_cost*middle),'Between-root surplus differs')
                # A common signed unit rescaling preserves the root set and order.
                scaled=name+'-scaled';factor=-8.
                result=await runner.CDGExecutionSession(None,'synthetic-reuse',scaled).execute({k:v*factor for k,v in inputs.items()},cdg=graph)
                require(result['status']=='completed','Scaled reuse execution failed')
                np.testing.assert_array_equal(np.load(Path(tmp)/scaled/'roots/out_lower_root.npy'),low)
                np.testing.assert_array_equal(np.load(Path(tmp)/scaled/'roots/out_upper_root.npy'),high)
                reports.append(dict(domain=name,synthetic_cases=n,both_roots_exact=True,independent_model_residuals_exact=True,signed_coefficient_rescaling_verified=True))
        finally:runner.RUNS_DIR=previous
    require(selected(approved)[0]==graph,'Stored graph changed during reuse execution')
    return dict(passed=True,approved=approved,catalog_mutations=0,version_id=imported['version_id'],graph_sha256=imported['graph_sha256'],
        reused_atoms=1,new_atoms=0,domains=reports,full_runner_cases=4,distinct_synthetic_cases=256,
        contracts=[
            'Mechanics adapter supplies coefficients in metres/seconds^2, metres/second and metres; outputs are times in seconds. Constant acceleration and zero initial height are constructed assumptions, not inferred by the solver.',
            'Cost adapter supplies coefficients in currency/quantity^2, currency/quantity and currency; roots are break-even quantities for constructed quadratic revenue and linear costs.',
            'Both algebraic roots are returned. A consumer chooses any physically admissible root using its own application constraints. No hidden event selection or rounding is applied.',
            'Numerical core receives finite real coefficients with identical shapes and consistent polynomial units; physical/economic validity is caller-owned.',
        ],limitations=['Synthetic constructed transfer evidence, not empirical motion validation, economic prediction or financial advice.',
            'No complex roots, linear fallback, arbitrary physical model, unit inference or implicit root selection.',
            'Provisioned in-process runner only; no HTTP, clean-install or performance claim.'],
        validator_sha256=sha(__file__),checker_sha256=sha(ROOT/'scripts/check_quadratic_revision.py'),runner_sha256=sha(ROOT/'sciona/visualizer/runner.py'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--approved',action='store_true');args=parser.parse_args()
    report=asyncio.run(validate(args.approved))
    name='quadratic_revision_served_reuse.json' if args.approved else 'quadratic_revision_reuse.json'
    (ROOT/'docs/reviews'/name).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,approved=args.approved,domains=2,synthetic_cases=256,runner_cases=4)))
