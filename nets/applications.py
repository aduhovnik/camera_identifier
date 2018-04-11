import keras
from .mobilenet_custom import MobileNet
from .densenet_keras import DenseNet201
from .densenet import DenseNetImageNet161, DenseNetImageNet264
#from .resnet import ResnetBuilder
from keras.models import Model, load_model
from keras.regularizers import l2
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten, Input, Concatenate

_ALLOWED_FE = {
    'resnet50': keras.applications.resnet50.ResNet50,
    'inception_v3': keras.applications.inception_v3.InceptionV3,
    #'nasnet_large': keras.applications.nasnet.NASNetLarge,
    #'nasnet_mobile': keras.applications.nasnet.NASNetMobile,
    'xception': keras.applications.xception.Xception,
    'mobilenet': MobileNet,
    'densenet161': DenseNetImageNet161,
    'densenet201': DenseNet201,
    'densenet264': DenseNetImageNet264,
    #'mobilenet4x4': MobileNet4x4
}


def build_model(fe, input_size=(512, 512), hinge=False, data_format='channels_last'):
    if fe not in _ALLOWED_FE.keys():
        raise ValueError('{} feature extractor is not supported'.format(fe))
    fe_class = _ALLOWED_FE[fe]
    weights = 'imagenet'
    if fe.startswith('nasnet') or fe == 'mobilenet4x4':
        weights = None
    aux_input = Input(shape=[64], name='aux_input')
    base_model = fe_class(weights=weights, include_top=False)
    #base_model = load_model('mobilenet4x4.h5', custom_objects={'relu6': keras.applications.mobilenet.relu6, 'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D})
#   for layer in base_model.layers:
#       layer.trainable = False
    # print(base_model.summary())
    x = base_model.output
    #import ipdb; ipdb.set_trace()
    #if not fe.startswith('densenet'):
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Concatenate()([x, aux_input])
    x = Dense(2048, activation='relu')(x)
    if hinge:
        predictions = Dense(10, W_regularizer=l2(0.01), activation='linear')(x)
    else:
        predictions = Dense(10, activation='softmax')(x)
    model = Model(inputs=[base_model.input, aux_input], outputs=predictions)
    return model
    # return base_model
