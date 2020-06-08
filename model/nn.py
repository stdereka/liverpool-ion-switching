from tensorflow.keras.layers import Conv1D, Input, Dense, Add, Multiply
import tensorflow_addons as tfa
from tensorflow.keras import models
import tensorflow as tf


def conv_block(x: tf.Tensor, filters: int, kernel_size: int):
    """
    Implements convolution block with residual connection.
    :param x: Input tensor.
    :param filters: Number of filters in convolution layer.
    :param kernel_size: Filter size.
    :return: Output tensor.
    """
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


def wave_block(x: tf.Tensor, filters: int, kernel_size: int, n: int):
    """
    Implements wavenet block.
    :param x: Input tensor.
    :param filters: Number of kernels.
    :param kernel_size: Filter size.
    :param n: Number of dilation rates for convolutions.
    :return: Output tensor.
    """
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


def model_1(x: tf.Tensor):
    """
    Base wavenet model without input and output layers.
    :param x: Input tensor.
    :return: Output tensor.
    """
    x = wave_block(x, 16, 3, 12)
    x = wave_block(x, 32, 3, 8)
    x = wave_block(x, 64, 3, 4)
    x = wave_block(x, 128, 3, 1)
    return x


def model_2(x):
    """
    Wavenet model with more kernels.
    :param x: Input tensor.
    :return: Output tensor.
    """
    x = wave_block(x, 32, 3, 12)
    x = wave_block(x, 64, 3, 8)
    x = wave_block(x, 128, 3, 4)
    x = wave_block(x, 256, 3, 1)
    return x


def model_3(x):
    """
    Wavenet model with residual convolution blocks.
    :param x: Input tensor.
    :return: Output tensor.
    """
    x = conv_block(x, 16, 3)
    x = wave_block(x, 16, 3, 12)
    x = conv_block(x, 32, 3)
    x = wave_block(x, 32, 3, 8)
    x = conv_block(x, 64, 3)
    x = wave_block(x, 64, 3, 4)
    x = conv_block(x, 128, 3)
    x = wave_block(x, 128, 3, 1)
    return x


# Dict for storing different versions of models
MODELS = {
    1: model_1,
    2: model_2,
    3: model_3
}


def get_model(version: int, shape: tuple, n_classes: int, loss: tf.losses.Loss, opt: tf.optimizers.Optimizer):
    """
    Builds and compiles wavenet model of given version.
    :param version: Model version. Must be 1, 2 or 3.
    :param shape: Input layer shape. Should be (batch_size, signal_length, number_of_channels).
    :param n_classes: Number of classes to predict.
    :param loss: Loss function to optimize.
    :param opt: Optimizer to use.
    :return: model - compiled model, prepared to training.
    """
    assert version in [1, 2, 3]
    inp = Input(shape=shape)
    x = MODELS[version](inp)
    out = Dense(n_classes, activation='softmax', name='out')(x)
    model = models.Model(inputs=inp, outputs=out)
    opt = tfa.optimizers.SWA(opt)
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    return model
