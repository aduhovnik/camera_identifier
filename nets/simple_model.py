from keras.regularizers import l2
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, Activation, MaxPooling2D, Dropout, Flatten

DROPOUT_VALUE = 0.5


def build_model(input_size, hinge=False):
    model = Sequential()

    model.add(Conv2D(filters=64, kernel_size=3, strides=1,
                     padding='same',
                     input_shape=[input_size[0], input_size[1], 3]))

    model.add(Activation('relu'))

    model.add(Conv2D(filters=64, kernel_size=3, strides=1,
                     padding='same'))

    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(DROPOUT_VALUE))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))

    model.add(Conv2D(filters=128, kernel_size=3, strides=1,
                     padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=128, kernel_size=3, strides=1,
                     padding='same'))

    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(DROPOUT_VALUE))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))

    model.add(Conv2D(filters=256, kernel_size=3, strides=1,
                     padding='same'))

    model.add(Activation('relu'))

    model.add(Conv2D(filters=256, kernel_size=3, strides=1,
                     padding='same'))

    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(DROPOUT_VALUE))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))

    model.add(Conv2D(filters=512, kernel_size=3, strides=1,
                     padding='same'))

    model.add(Activation('relu'))

    model.add(Conv2D(filters=512, kernel_size=3, strides=1,
                     padding='same'))

    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(DROPOUT_VALUE))
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))

    model.add(GlobalAveragePooling2D())

    # model.add(Flatten())
    model.add(Dense(1024))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    if hinge:
        predictions = Dense(10, W_regularizer=l2(0.01), activation='linear')(x)
    else:
        predictions = Dense(10, activation='softmax')(x)
    return model
