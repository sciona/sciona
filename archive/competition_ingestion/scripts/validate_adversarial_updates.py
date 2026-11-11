"""Pinned update-tail AST execution with NumPy ops and independent synthetic oracles."""
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import warnings

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.adversarial_updates import configuration,update


def main():
    pins=json.loads((ROOT/'docs/reviews/competition_adversarial_source_pins.json').read_text())
    blocks={}
    for branch,file in [('non_targeted','attack_iter.py'),('targeted','target_attack.py')]:
        source=next(s for s in pins['sources'] if s['branch']==branch)
        path=Path('/private/tmp/sciona_adversarial_source')/branch/file
        assert hashlib.sha256(path.read_bytes()).hexdigest()==next(f['sha256'] for f in source['files'] if f['path']==file)
        for fn in ast.parse(path.read_text()).body:
            if not isinstance(fn,ast.FunctionDef) or not fn.name.startswith('graph'):continue
            initial=[];tail=[];started=False
            for node in fn.body:
                text=ast.unparse(node)
                if text.startswith('noise = tf.gradients'):started=True;continue
                if started:
                    if text.startswith('i ='):break
                    tail.append(node)
                elif isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('eps','alpha','momentum','num_iter'):
                    initial.append(node)
            assert len(tail)==(4 if branch=='non_targeted' else 5)
            key='non_targeted' if branch=='non_targeted' else 'targeted_'+fn.name.split('_')[1]
            blocks[key]=compile(ast.Module(body=initial+tail,type_ignores=[]),'<pinned-update-tail>','exec')
    tf=SimpleNamespace(reduce_mean=lambda a,axes,keep_dims:np.mean(a,axis=tuple(axes),keepdims=keep_dims),
                       abs=np.abs,sign=np.sign,clip_by_value=np.clip,reshape=np.reshape,round=np.round,
                       contrib=SimpleNamespace(keras=SimpleNamespace(backend=SimpleNamespace(std=np.std))))
    rng=np.random.default_rng(321);cases=0
    for mode in ('non_targeted','targeted'):
        for epsilon in (0.,7.99,8.,16.):
            for dtype in (np.float32,np.float64):
                for momentum in (0.,1.):
                    x=rng.uniform(-.9,.9,(2,3,4,3)).astype(dtype)
                    gradient=rng.normal(size=x.shape).astype(dtype);previous=rng.normal(size=x.shape).astype(dtype)
                    lower=np.clip(x-2*epsilon/255,-1,1);upper=np.clip(x+2*epsilon/255,-1,1)
                    namespace={'tf':tf,'FLAGS':SimpleNamespace(max_epsilon=epsilon,num_iter=10,momentum=momentum,batch_size=2),
                               'x':x.copy(),'noise':gradient.copy(),'grad':previous.copy(),'x_min':lower,'x_max':upper}
                    config=configuration(mode,epsilon)
                    exec(blocks[config['branch']],namespace)
                    actual,noise=update(x,gradient,previous,lower,upper,mode=mode,epsilon=epsilon,momentum=momentum)
                    np.testing.assert_array_equal(actual,namespace['x']);np.testing.assert_array_equal(noise,namespace['noise'])
                    assert np.all(actual>=lower) and np.all(actual<=upper);cases+=1
    # Independent symmetric-gradient oracle: both divisors equal one exactly.
    x=np.zeros((1,1,2,3));g=np.array([-1,1,-1,1,-1,1.]).reshape(x.shape)
    for mode,epsilon,denominator in [('non_targeted',16.,10),('targeted',16.,12),('targeted',4.,28)]:
        actual,noise=update(x,g,x,np.full_like(x,-1),np.full_like(x,1),mode=mode,epsilon=epsilon,momentum=0.)
        np.testing.assert_array_equal(noise,g)
        np.testing.assert_allclose(actual,(1 if mode=='non_targeted' else -1)*2*epsilon/255/denominator*g,rtol=0,atol=0)
    # Offset does not change population std=1: exact half ties and clipping.
    for offset,pair in [(.5,[0.,2.]),(-.5,[-2.,0.]),(2.,[1.,2.]),(-2.,[-2.,-1.])]:
        actual,noise=update(x,g+offset,x,np.full_like(x,-1),np.full_like(x,1),
                            mode='targeted',epsilon=16.,momentum=0.)
        expected=np.array(pair*3).reshape(x.shape)
        np.testing.assert_array_equal(noise,g+offset)
        np.testing.assert_allclose(actual,-(2.*16./255./12)*expected,rtol=0,atol=0)
    bounded,_=update(x,g,x,np.full_like(x,-.001),np.full_like(x,.001),
                     mode='targeted',epsilon=16.,momentum=0.)
    np.testing.assert_array_equal(bounded,-.001*g)
    assert configuration('targeted',7.99)['models']==2 and configuration('targeted',8.)['models']==5
    # Source has no stabilizer. Zero gradients produce NaNs in both branch types.
    rejected=0
    for mode,epsilon in [('non_targeted',16.),('targeted',4.),('targeted',16.)]:
        ns={'tf':tf,'FLAGS':SimpleNamespace(max_epsilon=epsilon,num_iter=10,momentum=1.,batch_size=1),
            'x':x.copy(),'noise':x.copy(),'grad':x.copy(),'x_min':np.full_like(x,-1),'x_max':np.full_like(x,1)}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning);exec(blocks[configuration(mode,epsilon)['branch']],ns)
        assert np.isnan(ns['x']).all()
        try:update(x,x,x,np.full_like(x,-1),np.full_like(x,1),mode=mode,epsilon=epsilon)
        except ValueError:rejected+=1
        else:raise AssertionError('undefined gradient accepted')
    paths=['sciona/adversarial_updates.py','scripts/validate_adversarial_updates.py','docs/reviews/competition_adversarial_source_pins.json']
    report={'format':'adversarial-updates-validation.v1','result':'passed',
            'checks':{'source_tail_cases':cases,'independent_symmetric_oracles':3,'rounding_tie_and_step_clip_oracles':4,'independent_projection_oracle':True,'targeted_epsilon_boundary':True,
                      'source_nan_and_explicit_rejection_cases':rejected},
            'oracle_scope':'Pinned source assignment AST executed with NumPy equivalents for TensorFlow array ops; supplied synthetic gradients. Does not establish TensorFlow binary reduction parity.',
            'rounding_reference':'https://www.tensorflow.org/api_docs/python/tf/math/round',
            'adaptation':'Reject zero/nonfinite normalization instead of source NaN propagation, without adding an epsilon stabilizer.',
            'limits':'Update rules only; neural model gradients, label inference, ensemble loss, full iteration and serialized publication remain unverified.',
            'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}}
    (ROOT/'docs/reviews/competition_adversarial_updates.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':main()
