import keras
import keras.backend as K
# keras.backend.set_image_data_format('channels_first')
from keras.models import load_model, Model
from keras.layers import Activation, Dense, GlobalAveragePooling2D, Dropout

_ALLOWED_FE = ['resnet18', 'resnet34', 'se-resnet50']


def build_model(fe, input_size, hinge=False, layers_to_delete=3):
    if fe not in _ALLOWED_FE:
        raise ValueError(
            '{} feature extractor is not supported'.format(fe))
    model = load_model("{}_imagenet.h5".format(fe))
    x = model.layers[-(layers_to_delete + 1)].output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)
    model = Model(input=model.input, output=x)
    return model
