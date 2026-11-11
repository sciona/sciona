"""Compare corrected complete sweeps against a covariance-inverse reference."""
import argparse
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sciona.connectomics_source import source_namespace


def reference_filter(x,kind,threshold,tuned):
    t=np.arange(len(x))
    if kind=='f1': out=x+x[(t+1)%len(x)]+x[(t-1)%len(x)]
    elif kind=='f2':out=x+x[(t-1)%len(x)]+.8*x[(t-2)%len(x)]+.4*x[(t-3)%len(x)]
    elif kind=='f3':out=x+x[(t+1)%len(x)]+x[(t+2)%len(x)]+x[(t-1)%len(x)]
    else:out=x+x[(t+1)%len(x)]+x[(t+2)%len(x)]+x[(t+3)%len(x)]
    out=out[1:]-out[:-1]
    out[out<threshold]=0
    if tuned:out=out**.9
    sums=out.sum(axis=1)
    if tuned:sums=sums+.5*np.concatenate((sums[-1:],sums[:-1]))
    maximum=sums.max()
    for i,total in enumerate(sums):
        if not total:out[i]=1;continue
        power=1.
        if tuned:
            ratio=total/maximum
            if kind in ('f1','f2'):power=1.9 if .05<ratio<.23 else 1.6 if ratio<.75 else 1.4
            elif kind=='f3':power=1.9 if .04<ratio<.22 else 1.7 if ratio<.75 else 1.5
            else:power=1.9 if .08<ratio<.22 else 1.5
        weighted=(out[i]+1)**(1+1./total)
        out[i]=weighted**power if tuned else weighted
    return out


def inverse_covariance(x):
    centered=x.copy()
    centered-=centered.mean(axis=0)
    _,singular,v=np.linalg.svd(centered.astype(np.float64),full_matrices=False)
    variance=singular**2/len(centered)
    k=int(.8*x.shape[1]);noise=variance[k:].mean()
    covariance=(v[:k].T*np.maximum(variance[:k]-noise,0))@v[:k]+noise*np.eye(x.shape[1])
    return np.linalg.inv(covariance)


def scale(x):
    out=x.copy();np.fill_diagonal(out,out.min())
    return (out-out.min())/(out.max()-out.min())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    ns=source_namespace(args.source)
    tree=ast.parse((args.source/'code/PCA.py').read_bytes())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_tuned_inference')
    thresholds=next(ast.literal_eval(n.value) for n in fn.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='t' for t in n.targets))
    x=np.asfortranarray(np.random.default_rng(1729).uniform(0,.8,(240,10)),dtype=np.float32)
    legacy=ns['PCA'];fit_inputs=[]
    class CountPCA(legacy):
        def fit(self,values,*a,**kw):
            fit_inputs.append(hashlib.sha256(values.tobytes()).hexdigest())
            return super().fit(values,*a,**kw)
    ns['PCA']=CountPCA
    checks=[]
    for tuned in (False,True):
        kinds=['f1','f2','f3','f4'] if tuned else ['f1','f2']
        coefficients=[1.,.9,.01,.7] if tuned else [1.,.9]
        aggregate=np.zeros((10,10));weight=0;preprocessing_error=0.
        for threshold in thresholds:
            for kind,coefficient in zip(kinds,coefficients):
                ref=reference_filter(x,kind,threshold,tuned)
                actual=ns['tuned_filter' if tuned else 'simple_filter'](x,LP=kind,threshold=threshold)
                # Indexed gathers use a different memory layout/reduction order
                # than the source's Fortran arrays. Retained support must be
                # exact; float32 weighted amplitudes allow rounding only.
                np.testing.assert_array_equal(ref==1,actual==1)
                np.testing.assert_allclose(ref,actual,rtol=2e-6,atol=2e-6)
                preprocessing_error=max(preprocessing_error,float(np.max(abs(ref-actual))))
                aggregate-=coefficient*inverse_covariance(ref)
                weight+=coefficient
        expected=scale(aggregate/weight)
        fit_inputs.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            actual=ns['make_tuned_inference' if tuned else 'make_simple_inference'](x)
        np.testing.assert_allclose(actual,expected,rtol=3e-5,atol=3e-5)
        assert len(fit_inputs)==120*len(kinds) and len(set(fit_inputs))>len(kinds)*10
        checks.append({'mode':'tuned' if tuned else 'simple','pca_fits':len(fit_inputs),
            'distinct_preprocessed_inputs':len(set(fit_inputs)),
            'max_abs_covariance_reference_error':float(np.max(abs(actual-expected))),
            'preprocessing_support_exact':True,
            'max_abs_preprocessing_error':preprocessing_error,
            'preprocessing_rtol':2e-6,'preprocessing_atol':2e-6})
    args.output.write_text(json.dumps({'checks':checks,'grid_entries':len(thresholds),'unique_thresholds':len(set(thresholds)),
        'scope':'Complete corrected simple/tuned sweeps, scalar-indexed preprocessing reference and float64 SVD/covariance inverse. Historical source PCA runs float32; no original buggy-output parity claim.',
        'runtime_sha256':hashlib.sha256((Path(__file__).resolve().parents[1]/'sciona/connectomics_source.py').read_bytes()).hexdigest(),
        'validator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    print(json.dumps(checks))


if __name__=='__main__':main()
