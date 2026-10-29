"""Draft training factory integrating reusable repetition and score sampling.

The caller owns loading and epoch shuffling. This connects the source-validated
sample algorithms without claiming historical DataLoader worker scheduling.
"""
import random
import numpy as np

from sciona.dsb_components import _integer
from sciona.dsb_sampling import detector_epoch_length
from sciona.dsb_decomposed_sampling import detector_sampling_table,classifier_training_sample
from sciona.dsb_training import detector_indexed_training_sample


def _order(values,limit,name):
    result=tuple(_integer(value,name,0) for value in values)
    if len(result)!=limit or set(result)!=set(range(limit)):
        raise ValueError(name+' must be a permutation of the complete epoch')
    return result


class TrainingBatchFactory:
    """Reopen each pass while preserving caller-supplied random stream state.

    Detector and classifier volumes are already source-preprocessed CZYX arrays.
    Proposals and known indicators must be filtered together before this boundary.
    Explicit orders include every sample, with a retained final partial batch.
    """
    def __init__(self,detector_volumes,boxes_by_image,eligible_image_indices,
                 classifier_volumes,proposals_by_case,known_by_case,case_labels,
                 *,detector_order,classifier_order,detector_batch_size,classifier_batch_size,
                 topk=5,detector_crop_size=128,classifier_crop_size=96,
                 resolution=1.,random_fraction=.3,numpy_rng=None,label_rng=None):
        self.detector_volumes=detector_volumes
        self.boxes_by_image=boxes_by_image
        self.eligible=tuple(eligible_image_indices)
        self.classifier_volumes=classifier_volumes
        self.proposals=proposals_by_case
        self.known=known_by_case
        self.labels=np.asarray(case_labels,dtype=np.float32)
        count=len(classifier_volumes)
        if (len(detector_volumes)!=len(boxes_by_image) or not count
                or len(proposals_by_case)!=count or len(known_by_case)!=count
                or self.labels.shape!=(count,) or not np.all((self.labels==0)|(self.labels==1))):
            raise ValueError('aligned volumes, annotations and binary case labels required')
        self.table=detector_sampling_table(boxes_by_image,resolution)
        self.orders={'detector':_order(detector_order,detector_epoch_length(self.table,random_fraction),'detector_order'),
                     'classifier':_order(classifier_order,count,'classifier_order')}
        self.batch_sizes={'detector':_integer(detector_batch_size,'detector_batch_size'),
                          'classifier':_integer(classifier_batch_size,'classifier_batch_size')}
        self.topk=_integer(topk,'topk')
        self.detector_crop_size=_integer(detector_crop_size,'detector_crop_size')
        self.classifier_crop_size=_integer(classifier_crop_size,'classifier_crop_size')
        self.rng=np.random.RandomState() if numpy_rng is None else numpy_rng
        self.label_rng=random.Random() if label_rng is None else label_rng

    def __call__(self,task,profile):
        if task not in self.orders:raise ValueError('unknown training task')
        pending=[]
        for index in self.orders[task]:
            if task=='detector':
                sample=detector_indexed_training_sample(index,self.detector_volumes,
                    self.boxes_by_image,self.table,self.eligible,rng=self.rng,label_rng=self.label_rng,
                    crop_size=self.detector_crop_size,**profile)
            else:
                images,coords,known,_=classifier_training_sample(self.classifier_volumes[index],
                    self.proposals[index],self.known[index],topk=self.topk,crop_size=self.classifier_crop_size,
                    rng=self.rng,**profile)
                sample=(images,coords,known,self.labels[index:index+1])
            pending.append(sample)
            if len(pending)==self.batch_sizes[task]:
                yield tuple(np.stack(items) for items in zip(*pending))
                pending=[]
        if pending:yield tuple(np.stack(items) for items in zip(*pending))
