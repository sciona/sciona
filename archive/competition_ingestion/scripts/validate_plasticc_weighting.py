"""Synthetic source parity and independent fold/weight/loss checks."""
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona import plasticc_weighting as runtime


def validate(root):
    settings=dict(num_folds=3,fold_random_state=42,redshift_weighting_group_key=None,
        redshift_weighting_min_redshift=.1,redshift_weighting_max_redshift=1.,
        redshift_weighting_num_bins=4,redshift_weighting_min_bin_count=2,
        redshift_weighting_redshift_key='host_specz')
    ns=dict(vars(runtime),settings=settings)
    pins=json.loads((ROOT/'docs/reviews/competition_plasticc_source_pins.json').read_text())
    for name in ['avocado/classifier.py','avocado/dataset.py']:
        p=root/name
        assert hashlib.sha256(p.read_bytes()).hexdigest()==pins['files'][name]
        tree=ast.parse(p.read_text())
        if 'classifier' in name:
            nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in
                {'evaluate_weights_flat','evaluate_weights_redshift','weighted_multi_logloss'}]
        else:
            cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Dataset')
            nodes=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='label_folds']
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
    counts=dict(fold_assignments=0,augmentation_isolation=0,flat_weights=0,
                redshift_weights=0,independent_redshift=0,losses=0,loss_contributions=0,
                missing_prediction_rejection=0)
    rng=np.random.default_rng(3)
    for seed in range(6):
        labels=np.repeat([6,42,90],12)
        metadata=pd.DataFrame({'class':labels,'host_specz':np.where(labels==6,0,np.resize([.01,.1,.2,.5,1.,2.],36)),
                               'group':np.resize([0,1],36)},index=['synthetic-%d'%i for i in range(36)])
        dataset=SimpleNamespace(name='synthetic',metadata=metadata)
        for augmented in (False,True):
            if augmented:
                aug=metadata.iloc[::2].copy()
                aug['reference_object_id']=aug.index
                aug.index=aug.index+'-aug'
                dataset.metadata=pd.concat([metadata,aug])
            folds=runtime.label_folds(dataset,random_state=seed,settings=settings)
            pd.testing.assert_series_equal(folds,ns['label_folds'](dataset,random_state=seed))
            assert set(folds)=={0,1,2}
            if augmented:
                for key,row in aug.iterrows(): assert folds[key]==folds[row.reference_object_id]
                counts['augmentation_isolation']+=1
            counts['fold_assignments']+=1
            for class_weights in [None,{6:1.,42:2.,90:0.}]:
                flat=runtime.evaluate_weights_flat(dataset,class_weights)
                pd.testing.assert_series_equal(flat,ns['evaluate_weights_flat'](dataset,class_weights))
                for label in np.unique(labels):
                    np.testing.assert_allclose(flat[dataset.metadata['class']==label].sum(),
                        len(dataset.metadata)*(1 if class_weights is None else class_weights[label]))
                counts['flat_weights']+=1
                for group in [None,'group']:
                    weights=runtime.evaluate_weights_redshift(dataset,class_weights,group_key=group,settings=settings)
                    pd.testing.assert_series_equal(weights,ns['evaluate_weights_redshift'](dataset,class_weights,group_key=group))
                    # Independent row-count oracle with source left-side boundary inclusion.
                    md=dataset.metadata
                    edges=np.logspace(-1,0,5)[1:-1]
                    def bin_for(z): return 0 if z<=1e-99 else 1+sum(z>x for x in edges)
                    keys=[(row[group] if group else 0,bin_for(row.host_specz),row['class']) for _,row in md.iterrows()]
                    from collections import Counter
                    totals=Counter(keys)
                    extragal_classes=[label for label in np.unique(labels) if sum(k[1]>0 and k[2]==label for k in keys)>sum(k[1]==0 and k[2]==label for k in keys)]
                    scale=sum(k[1]>0 and v>1e-4*len(md) for k,v in totals.items())/len(extragal_classes)
                    expected=[len(md)/max(totals[k],2)/(scale if k[2] in extragal_classes else 1)*(1 if class_weights is None else class_weights[k[2]]) for k in keys]
                    np.testing.assert_allclose(weights,expected)
                    counts['redshift_weights']+=1; counts['independent_redshift']+=1
                predictions=rng.uniform(.01,1,(len(dataset.metadata),3)); predictions/=predictions.sum(axis=1)[:,None]
                predictions=pd.DataFrame(predictions,index=dataset.metadata.index,columns=[6,42,90])
                truth=dataset.metadata['class']
                object_weights=pd.Series(rng.uniform(.1,2,len(truth)),index=truth.index)
                loss=runtime.weighted_multi_logloss(truth,predictions,object_weights,class_weights)
                np.testing.assert_allclose(loss,ns['weighted_multi_logloss'](truth,predictions,object_weights,class_weights))
                expected=0; denom=0
                for label in np.unique(labels):
                    cw=1 if class_weights is None else class_weights[label]
                    mask=truth==label
                    expected+=cw*np.average(-np.log(predictions.loc[mask,label]),weights=object_weights[mask]); denom+=cw
                np.testing.assert_allclose(loss,expected/denom)
                contributions=runtime.weighted_multi_logloss(truth,predictions,object_weights,class_weights,True)
                np.testing.assert_allclose(contributions.sum(),loss)
                counts['losses']+=1; counts['loss_contributions']+=1
    try: runtime.weighted_multi_logloss(truth,predictions.drop(columns=42))
    except runtime.AvocadoException: counts['missing_prediction_rejection']+=1
    else: raise AssertionError('Missing class accepted')
    paths=[Path(runtime.__file__),Path(__file__),ROOT/'docs/reviews/competition_plasticc_source_pins.json',ROOT/'docs/licenses/Avocado-MIT.txt']
    return dict(status='passed',approved=False,checks=counts,
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        scope='Fold assignment, flat/redshift weights and weighted loss; synthetic settings and inputs',
        remaining=['Pinned default settings review','LightGBM compatibility and actual fitting','Dataset orchestration and full graph'])

if __name__=='__main__':
    result=validate(Path(sys.argv[1]) if len(sys.argv)>1 else Path('/private/tmp/sciona_plasticc_source'))
    (ROOT/'docs/reviews/competition_plasticc_weighting.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['checks'],sort_keys=True))
