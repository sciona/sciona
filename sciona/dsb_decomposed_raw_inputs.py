"""Prepare explicit raw runtime collections for DSB training and validation.

Proposal arrays are caller-supplied detector outputs in the resulting preprocessed
ZYX frame, as in the source classifier intake. No identity or file lookup occurs.
"""
import numpy as np
from sciona.dsb_components import _integer
from sciona.dsb_decomposed_preparation import prepare_training_volume
from sciona.dsb_decomposed_preparation import prepare_world_training_volume
from sciona.dsb_intake import annotate_classifier_proposals


def _prepare_record(record,classifier=False):
    required={'kind','preparation'}|({'proposals','case_label'} if classifier else set())
    if set(record)!=required:raise ValueError('explicit raw preparation record required')
    if record['kind']=='voxel':fn=prepare_training_volume
    elif record['kind']=='world':fn=prepare_world_training_volume
    else:raise ValueError('unknown annotation coordinate convention')
    prepared,boxes,_,_=fn(**record['preparation'])
    return prepared,boxes


def prepare_raw_collections(raw_collections,eligible_image_indices):
    """Apply source preparation and training proposal annotation to five roles."""
    keys={'training_detector','training_classifier','validation_detector',
          'validation_classifier','training_validation_classifier'}
    if set(raw_collections)!=keys or any(not raw_collections[k] for k in keys):
        raise ValueError('five explicit nonempty raw collections required')
    prepared={}
    for key in keys:
        classifier='classifier' in key
        volumes=[];boxes=[];proposals=[];labels=[]
        for record in raw_collections[key]:
            if classifier and record.get('case_label') not in (0,1):raise ValueError('binary case label required')
            volume,annotations=_prepare_record(record,classifier)
            volumes.append(volume);boxes.append(annotations)
            if classifier:
                proposals.append(np.asarray(record['proposals']))
                labels.append(record['case_label'])
        prepared[key]=dict(volumes=volumes,boxes=boxes)
        if classifier:prepared[key].update(proposals=proposals,labels=labels)
    detector=prepared['training_detector'];classifier=prepared['training_classifier']
    eligible=tuple(_integer(i,'eligible image index',0) for i in eligible_image_indices)
    if len(set(eligible))!=len(eligible) or any(i>=len(detector['volumes']) for i in eligible):
        raise ValueError('unique eligible indices within detector collection required')
    annotated=[annotate_classifier_proposals(p,b) for p,b in zip(classifier['proposals'],classifier['boxes'])]
    training=dict(detector_volumes=detector['volumes'],boxes_by_image=detector['boxes'],eligible_image_indices=eligible,
        classifier_volumes=classifier['volumes'],proposals_by_case=[p for p,_ in annotated],
        known_by_case=[k for _,k in annotated],case_labels=classifier['labels'])
    validation=dict(detector=prepared['validation_detector'],classifier_validation=prepared['validation_classifier'],
                    classifier_training_validation=prepared['training_validation_classifier'])
    return training,validation
