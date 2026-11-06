"""Synthetic layout-sensitive matmul diagnostic; does not alter runtime policy."""
import hashlib
import importlib.metadata as metadata
import json
import math
from pathlib import Path
import platform
import warnings

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def diagnose():
    cases=[]
    for rows in (2,8,9,16,19,32):
        left=np.random.default_rng(12).uniform(-1,1,(rows,9))
        right=np.random.default_rng(13).uniform(-1,1,(9,9))
        reference=np.array([[math.fsum(float(left[i,k])*float(right[k,j]) for k in range(9))
                             for j in range(9)] for i in range(rows)])
        for order in ('C','F'):
            a,b=np.array(left,order=order),np.array(right,order=order)
            for operation in ('matmul','einsum'):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    result=a@b if operation=='matmul' else np.einsum('ik,kj->ij',a,b,optimize=False)
                error=float(np.max(np.abs(result-reference)))
                finite=bool(np.isfinite(result).all())
                if not finite or error>1e-14:raise ValueError('Synthetic contraction failed independent reference')
                cases.append(dict(rows=rows,inner=9,columns=9,order=order,operation=operation,
                    inputs_finite=True,inputs_max_abs_below_one=True,output_finite=finite,
                    max_absolute_error_against_fsum=error,
                    warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in caught]))
    return dict(format='text-pair-matmul-diagnostic.v1',synthetic_only=True,catalog_mutations=0,
        approved=False,platform=dict(system=platform.system(),machine=platform.machine()),
        dependencies={n:metadata.version(n) for n in ('numpy','scipy','scikit-learn')},
        cases=cases,warning_cases=sum(bool(c['warnings']) for c in cases),
        independent_reference='Python math.fsum over nine float64 scalar products per output element.',
        conclusion='Inspect recorded layouts: warnings can arise in the finite ordinary-range contraction without any SVD call. Numerical agreement alone does not qualify the warning-producing path.',
        next_action='Qualify a reusable numerical decomposition with a warning-free contraction and complete downstream lifecycle comparisons; do not suppress warnings or change the shared environment based only on this diagnostic.',
        limits=['Only these synthetic shapes, values and installed runtime are tested.',
                'This diagnoses behavior, not the internal library cause; no universal accuracy or platform guarantee.',
                'Existing approved implementations and their historical evidence remain unchanged.'],
        diagnostic_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__=='__main__':
    result=diagnose()
    (ROOT/'docs/reviews/text_pair_matmul_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(cases=len(result['cases']),warning_cases=result['warning_cases'],
        maximum_absolute_error=max(c['max_absolute_error_against_fsum'] for c in result['cases']),approved=False)))
