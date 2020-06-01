from tensorflow.keras.layers import Conv1D, Input, Dense, Add, Multiply
import tensorflow_addons as tfa
from tensorflow.keras import models


def conv_block(x, filters, kernel_size):
    x = Conv1D(filters=filters,
               kernel_size=1,
               padding='same')(x)
    res_x = x
    x = Conv1D(filters=filters,
               kernel_size=kernel_size,
               padding='same', activation='relu')(x)
    x = Conv1D(filters=filters,
               kernel_size=kernel_size,
               padding='same', activation='relu')(x)
    x = Conv1D(filters=filters,
               kernel_size=kernel_size,
               padding='same', activation='relu')(x)
    res_x = Add()([res_x, x])
    return res_x


def wave_block(x, filters, kernel_size, n):
    dilation_rates = [2 ** i for i in range(n)]
    x = Conv1D(filters=filters,
               kernel_size=1,
               padding='same')(x)
    res_x = x
    for dilation_rate in dilation_rates:
        tanh_out = Conv1D(filters=filters,
                          kernel_size=kernel_size,
                          padding='same',
                          activation='tanh',
                          dilation_rate=dilation_rate)(x)
        sigm_out = Conv1D(filters=filters,
                          kernel_size=kernel_size,
                          padding='same',
                          activation='sigmoid',
                          dilation_rate=dilation_rate)(x)
        x = Multiply()([tanh_out, sigm_out])
        x = Conv1D(filters=filters,
                   kernel_size=1,
                   padding='same')(x)
        res_x = Add()([res_x, x])
    return res_x


def model_1(x):
    x = wave_block(x, 16, 3, 12)
    x = wave_block(x, 32, 3, 8)
    x = wave_block(x, 64, 3, 4)
    x = wave_block(x, 128, 3, 1)
    return x


def model_2(x):
    x = wave_block(x, 32, 3, 12)
    x = wave_block(x, 64, 3, 8)
    x = wave_block(x, 128, 3, 4)
    x = wave_block(x, 256, 3, 1)
    return x


def model_3(x):
    x = conv_block(x, 16, 3)
    x = wave_block(x, 16, 3, 12)
    x = conv_block(x, 32, 3)
    x = wave_block(x, 32, 3, 8)
    x = conv_block(x, 64, 3)
    x = wave_block(x, 64, 3, 4)
    x = conv_block(x, 128, 3)
    x = wave_block(x, 128, 3, 1)
    return x


MODELS = {
    1: model_1,
    2: model_2,
    3: model_3
}


def get_model(version, shape, n_classes, loss, opt):
    assert version in [1, 2, 3]
    inp = Input(shape=shape)
    x = MODELS[version](inp)
    out = Dense(n_classes, activation='softmax', name='out')(x)
    model = models.Model(inputs=inp, outputs=out)
    opt = tfa.optimizers.SWA(opt)
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    return model
