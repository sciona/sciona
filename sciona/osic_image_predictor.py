"""Source B5 inference geometry and fixed batch-size policy."""
from sciona.keras_bounded_prediction import make_predictor


def make_slice_predictor(model):
    if model.input_shape!=[(None,512,512,1),(None,4)] or model.output_shape!=(None,1):
        raise ValueError('OSIC image/metadata inputs and scalar slope output required')
    return make_predictor(model,batch_size=32)
