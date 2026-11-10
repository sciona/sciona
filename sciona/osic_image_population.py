"""Compose raw image population preparation while retaining source selection."""
import numpy as np

from sciona.dicom_planes import decode_plane
from sciona.osic_dicom import training_image
from sciona.osic_image_metadata import prepare_training
from sciona.osic_image_batches import validate_population
from sciona.osic_image_inference import predict


def prepare_training_population(table,encoded_planes,*,excluded_entities,time_coordinate,response_coordinate):
    """Fit source targets before exclusions; decode eligible groups in row order.

The caller supplies the private exclusion policy and ordered (name,bytes)
records per entity. Exclusions filter sampling eligibility, not the earlier
longitudinal fit. No source identifiers are embedded. Invalid eligible files
raise; shortened batches from failed source decodes are outside this contract.
"""
    if (not isinstance(excluded_entities,(list,tuple))
            or any(not isinstance(k,str) or not k for k in excluded_entities)
            or len(set(excluded_entities))!=len(excluded_entities)):
        raise ValueError('Explicit unique exclusion identities required')
    fitted=prepare_training(table,time_coordinate=time_coordinate,response_coordinate=response_coordinate)
    if not isinstance(encoded_planes,dict) or set(encoded_planes)!=set(fitted['entities']):
        raise ValueError('Image population must match the mapped training entities')
    keep=[i for i,k in enumerate(fitted['entities']) if k not in excluded_entities]
    if not keep:
        raise ValueError('At least one eligible training entity required')
    images=[]
    for i in keep:
        records=encoded_planes[fitted['entities'][i]]
        if (not isinstance(records,(list,tuple)) or not records
                or any(not isinstance(r,(list,tuple)) or len(r)!=2
                       or not isinstance(r[0],str) or not r[0] for r in records)
                or len({r[0] for r in records})!=len(records)):
            raise ValueError('Ordered unique named training images required')
        images.append(np.expand_dims(np.stack([training_image(payload) for _,payload in records]),axis=-1))
    metadata=fitted['metadata'][keep]
    slopes=fitted['slopes'][keep]
    validate_population(images,metadata,slopes)
    return dict(entities=[fitted['entities'][i] for i in keep],images=images,metadata=metadata,
        slopes=slopes,fits=[fitted['fits'][i] for i in keep],
        time_coordinate=time_coordinate,response_coordinate=response_coordinate)


def _inference_pixels(encoded):
    return decode_plane(encoded,read_rescale=False)['pixels']


def predict_from_encoded(baselines,queries,encoded_planes,predict_slices):
    """Decode selected files only, then reuse the qualified inference routing."""
    return predict(baselines,queries,encoded_planes,predict_slices,decode_selected=_inference_pixels)
