"""OSIC image slope adapter; requires the separately qualified legacy runtime.

The caller provisions efficientnet 1.1.0 and enables TF_USE_LEGACY_KERAS before
importing TensorFlow. No pretrained weights or input rescaling are introduced.
Only the selected B5 is constructed. This preserves topology, not the original
random stream consumed by constructing all eight unused/used architectures.
"""


def build_model():
    import efficientnet
    import efficientnet.tfkeras as efn
    from tensorflow import keras

    if efficientnet.__version__ != '1.1.0':
        raise ValueError('The qualified efficientnet 1.1.0 runtime is required')
    if not keras.layers.Layer.__module__.startswith('tf_keras.'):
        raise ValueError('The qualified legacy-Keras runtime is required')
    backbone = efn.EfficientNetB5(
        input_shape=(512, 512, 1), weights=None, include_top=False)
    image = keras.layers.Input(shape=(512, 512, 1))
    pooled = keras.layers.GlobalAveragePooling2D()(backbone(image))
    metadata = keras.layers.Input(shape=(4,))
    noisy = keras.layers.GaussianNoise(.2)(metadata)
    joined = keras.layers.Concatenate()([pooled, noisy])
    dropped = keras.layers.Dropout(.5)(joined)
    slope = keras.layers.Dense(1)(dropped)
    return keras.Model([image, metadata], slope)
