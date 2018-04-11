import keras
from keras import backend as K
#DATA_FORMAT = 'channels_first'
# keras.backend.set_image_data_format(DATA_FORMAT)
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model, Model
from keras.layers import Dense
from keras.optimizers import RMSprop, Adam, Nadam, SGD
from keras.losses import categorical_hinge
from keras.utils.generic_utils import CustomObjectScope
from utils.data_generator import get_generators
from nets.applications import build_model
#from nets.resnet_fusion import build_model
#from nets.pytorch_converted_networks import build_model

TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'
VAL_DIR = 'data/val_images'
TARGET_SIZE = (512, 512)
IMAGES_COUNT = 2750
BATCH_SIZE = 4
MODEL_NAME = 'densenet201'


def train_model(finetune=False):
    (train_generator, train_samples), (val_generator, val_samples) = get_generators(
        TRAIN_DIR, crop_size=TARGET_SIZE, batch_size=BATCH_SIZE, val_dir=VAL_DIR, multiple_data=False)
    #model = build_model(MODEL_NAME, TARGET_SIZE, hinge=False)
    custom_objects = {'relu6': keras.applications.mobilenet.relu6,
                      'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}
    model = load_model(
        'weights/densenet201_weights.03-0.20.h5', custom_objects=custom_objects, compile=False)

    if finetune:
        for layer in model.layers[:-2]:
            layer.trainable = False
        model.compile(optimizer='sgd',
                      loss='categorical_crossentropy', metrics=['acc'])
        model.fit_generator(train_generator,
                            validation_data=val_generator,
                            epochs=3,
                            verbose=1,
                            steps_per_epoch=int(train_samples / BATCH_SIZE),
                            validation_steps=int(val_samples / BATCH_SIZE),
                            use_multiprocessing=True,
                            workers=8,
                            )
        for layer in model.layers:
            layer.trainable = True
    # for layer in model.layers[:-2]:
    #    layer.trainable = False
    model.compile(optimizer=SGD(0.0005),
                  loss='categorical_crossentropy', metrics=['acc'])
    model.fit_generator(train_generator,
                        validation_data=val_generator,
                        epochs=300,
                        verbose=1,
                        steps_per_epoch=int(train_samples / BATCH_SIZE),
                        validation_steps=int(val_samples / BATCH_SIZE),
                        use_multiprocessing=True,
                        workers=12,
                        callbacks=[
                            ModelCheckpoint(
                                'weights/' + MODEL_NAME +
                                '_weights.{epoch:02d}-{loss:.2f}.h5'),
                            ReduceLROnPlateau(monitor='loss', factor=0.7, min_lr=1e-6,
                                              patience=2, verbose=1)
                        ]
                        )
    model.save('model.h5')

if __name__ == '__main__':
    train_model()
