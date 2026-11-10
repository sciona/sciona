"""Decode one monochrome DICOM plane without presentation transformations."""
from io import BytesIO

import numpy as np


def decode_plane(encoded, *, read_rescale):
    """Return stored integer pixels and optionally the numeric rescale metadata.

The caller owns private encoded bytes and decoded outputs. Single-frame,
single-sample MONOCHROME1/2 inputs only; no inversion, windowing, intensity
rescaling or identity metadata is returned. Decoding uses the provisioned
pydicom backend; codec qualification is separate from accepting this interface.
``read_rescale=False`` deliberately ignores rescale metadata, including missing
fields. With True both fields are required but are not applied to the pixels.
"""
    import pydicom
    if not isinstance(encoded,bytes) or not encoded or type(read_rescale) is not bool:
        raise ValueError('Nonempty encoded bytes and explicit rescale policy required')
    try:
        dataset=pydicom.dcmread(BytesIO(encoded))
        if (int(dataset.get('NumberOfFrames',1))!=1 or int(dataset.SamplesPerPixel)!=1
                or dataset.PhotometricInterpretation not in ('MONOCHROME1','MONOCHROME2')):
            raise ValueError('Unsupported plane geometry')
        pixels=dataset.pixel_array
        if pixels.ndim!=2 or not pixels.size or pixels.dtype.kind not in 'iu':
            raise ValueError('Integer monochrome plane required')
        rescale=None
        if read_rescale:
            rescale=dict(intercept=float(dataset.RescaleIntercept),slope=float(dataset.RescaleSlope))
        return dict(pixels=np.array(pixels,copy=True),rescale=rescale)
    except Exception:
        # Decoder diagnostics can contain caller-owned metadata. Do not embed it
        # in public exception text, logs or chained exceptions.
        raise ValueError('DICOM plane or requested rescale metadata could not be decoded') from None
