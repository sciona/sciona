"""Source training/inference policies composed with explicit DICOM decoding."""
from sciona.dicom_planes import decode_plane
from sciona.osic_image_preprocessing import training_plane,inference_plane


def training_image(encoded):
    decoded=decode_plane(encoded,read_rescale=True)
    return training_plane(decoded['pixels'],**decoded['rescale'])


def inference_image(encoded):
    decoded=decode_plane(encoded,read_rescale=False)
    return inference_plane(decoded['pixels'])
