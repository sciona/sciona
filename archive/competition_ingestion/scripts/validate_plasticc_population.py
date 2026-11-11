"""Integrated synthetic object-to-probability lifecycle, no catalog mutation."""
import hashlib
import json
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sciona.plasticc_population import Dataset,PopulationAugmentor
from sciona.plasticc_observations import AstronomicalObject
from sciona.plasticc_features import PlasticcFeaturizer,plasticc_start_time,plasticc_bands
from sciona.plasticc_training import LightGBMClassifier
from sciona.plasticc_predictions import create_kaggle_predictions
from sciona.plasticc_defaults import DEFAULTS


def validate():
    rng=np.random.default_rng(6); objects=[]
    for label in [6,42,52,62,95]:
        for i in range(3):
            times=np.linspace(plasticc_start_time,plasticc_start_time+1000,72)
            flux=1000*np.sin((times-plasticc_start_time)/(40+i*10)) + rng.normal(size=72)
            metadata=dict(object_id='synthetic-%d-%d'%(label,i),redshift=0 if label==6 else .2,
                host_specz=0 if label==6 else .2,host_photoz=0 if label==6 else .22,
                host_photoz_error=.02,galactic=label==6,ddf=False,mwebv=.01,ra=0.,decl=0.)
            metadata['class']=label
            objects.append(AstronomicalObject(metadata,pd.DataFrame(dict(time=times,
                band=np.resize(plasticc_bands,len(times)),flux=flux,flux_error=np.ones(len(times))))))
    dataset=Dataset.from_objects('synthetic',objects,chunk=0,num_chunks=1)
    assert list(dataset.metadata.index)==[o.metadata['object_id'] for o in objects]
    augmentor=PopulationAugmentor(np.array([[.1,.12,.02],[.4,.35,.04]]),seed=3,augment_retries=DEFAULTS['augment_retries'])
    population=augmentor.augment_dataset('synthetic',dataset,1)
    assert population.chunk==0 and population.num_chunks==1
    assert len(population)>len(dataset)
    originals=population.metadata.reference_object_id.isna()
    assert originals.sum()==len(dataset)
    folds=population.label_folds(num_folds=3)
    for key,row in population.metadata[~originals].iterrows(): assert folds[key]==folds[row.reference_object_id]
    featurizer=PlasticcFeaturizer()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        raw=population.extract_raw_features(featurizer,keep_models=True)
        selected=population.select_features(featurizer)
        assert raw.shape==(len(population),230) and selected.shape==(len(population),41)
        assert len(population.models)==len(population)
        pd.testing.assert_index_equal(selected.index,population.metadata.index)
        classifier=LightGBMClassifier('synthetic',featurizer)
        classifier.train(population,num_folds=3,n_estimators=60,n_jobs=1,
            min_child_weight=.01,min_child_samples=2,min_split_gain=0.,random_seed=4)
        predictions=population.predict(classifier)
        final=create_kaggle_predictions(population)
    assert population.classifier is classifier and population.predictions is predictions
    np.testing.assert_allclose(predictions.sum(axis=1),1)
    np.testing.assert_allclose(final.sum(axis=1),1)
    assert np.isfinite(final.to_numpy()).all() and (final.to_numpy()>=0).all()
    assert (final.loc[population.metadata.galactic,[42,52,62,95]]==0).all().all()
    assert (final.loc[~population.metadata.galactic,6]==0).all()
    # Replay augmentation from fresh seed; GP cache must not change random draws.
    replay=PopulationAugmentor(np.array([[.1,.12,.02],[.4,.35,.04]]),seed=3,augment_retries=DEFAULTS['augment_retries']).augment_dataset('synthetic',dataset,1)
    pd.testing.assert_frame_equal(population.metadata,replay.metadata)
    for left,right in zip(population.objects,replay.objects): pd.testing.assert_frame_equal(left.observations,right.observations)
    paths=[ROOT/'sciona'/('plasticc_'+name+'.py') for name in ['population','defaults','observations','augmentation','features','weighting','training','predictions']]
    paths += [Path(__file__),ROOT/'docs/reviews/competition_plasticc_defaults.json']
    return dict(status='passed',approved=False,checks=dict(original_objects=len(dataset),augmented_objects=len(population)-len(dataset),
        raw_features=raw.shape[1],selected_features=selected.shape[1],trained_folds=len(classifier.classifiers),
        normalized_output_rows=len(final),augmentation_replays=1),
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        scope='Integrated actual synthetic augmentation/GP/features/folds/train/predict/postprocess',
        limitations=['Synthetic training capacity overrides; no accuracy claim','Serialized payload/graph and publication review outstanding'])

if __name__=='__main__':
    report=validate()
    (ROOT/'docs/reviews/competition_plasticc_population.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks'],sort_keys=True))
