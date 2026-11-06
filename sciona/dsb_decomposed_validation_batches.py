"""In-memory validation loaders with source sampling and proposal annotation."""
import random
import numpy as np

from sciona.dsb_components import _integer
from sciona.dsb_decomposed_sampling import detector_sampling_table
from sciona.dsb_intake import annotate_classifier_proposals
from sciona.dsb_decomposed_evaluation_samples import classifier_validation_sample, detector_validation_sample


class ValidationBatchFactory:
    """Reopen three distinct validation collections without training augmentation.

    All volumes/boxes/proposals share the preprocessed coordinate frame. Detector
    size repeats remain; no random crop extras are added. Classifier proposals are
    filtered, suppressed and annotated once before score-ordered validation crops.
    """
    def __init__(self,validation_arrays,*,detector_batch_size,classifier_batch_size,
                 topk=5,detector_crop_size=128,classifier_crop_size=96,resolution=1.,
                 numpy_rng=None,label_rng=None):
        keys={'detector','classifier_validation','classifier_training_validation'}
        if set(validation_arrays)!=keys:raise ValueError('three explicit validation collections required')
        self.arrays=validation_arrays
        detector=validation_arrays['detector']
        if set(detector)!={'volumes','boxes'} or len(detector['volumes'])!=len(detector['boxes']):
            raise ValueError('aligned detector validation volumes and boxes required')
        self.table=detector_sampling_table(detector['boxes'],resolution)
        if not len(self.table):raise ValueError('detector validation requires eligible targets')
        self.classifiers={}
        for key in ['classifier_validation','classifier_training_validation']:
            data=validation_arrays[key]
            if set(data)!={'volumes','proposals','boxes','labels'}:
                raise ValueError('explicit classifier validation collections required')
            count=len(data['volumes'])
            if not count or any(len(data[name])!=count for name in ['proposals','boxes','labels']):
                raise ValueError('classifier validation collections must align')
            labels=np.asarray(data['labels'])
            if labels.shape!=(count,) or not np.all((labels==0)|(labels==1)):
                raise ValueError('binary case labels required')
            annotated=[annotate_classifier_proposals(p,b) for p,b in zip(data['proposals'],data['boxes'])]
            self.classifiers[key]=(data['volumes'],annotated,labels)
        self.detector_batch_size=_integer(detector_batch_size,'detector_batch_size')
        self.classifier_batch_size=_integer(classifier_batch_size,'classifier_batch_size')
        self.topk=_integer(topk,'topk')
        self.detector_crop_size=_integer(detector_crop_size,'detector_crop_size')
        self.classifier_crop_size=_integer(classifier_crop_size,'classifier_crop_size')
        self.rng=np.random.RandomState() if numpy_rng is None else numpy_rng
        self.label_rng=random.Random() if label_rng is None else label_rng

    def __call__(self,task):
        if task=='detector':
            data=self.arrays[task]
            samples=(detector_validation_sample(data['volumes'][int(row[0])],row[1:],
                data['boxes'][int(row[0])],crop_size=self.detector_crop_size,
                rng=self.rng,label_rng=self.label_rng) for row in self.table)
            size=self.detector_batch_size
        elif task in self.classifiers:
            volumes,annotations,labels=self.classifiers[task]
            samples=(classifier_validation_sample(v,p,k,label,topk=self.topk,crop_size=self.classifier_crop_size)
                     for v,(p,k),label in zip(volumes,annotations,labels))
            size=self.classifier_batch_size
        else:raise ValueError('unknown validation task')
        pending=[]
        for sample in samples:
            pending.append(sample)
            if len(pending)==size:
                yield tuple(np.stack(items) for items in zip(*pending));pending=[]
        if pending:yield tuple(np.stack(items) for items in zip(*pending))
