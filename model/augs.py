"""
File with augmentations. In order to be compatible with DataGenerator, they should accept (X, y) and return modified
(X, y). Example of pipeline can be found on line 69. You can add your custom augmentations.
"""

import numpy as np


def flip(x, y):
    """
    Flips signal and groundtruth.
    :param x: Data.
    :param y: Labels.
    :return: Flipped x and y.
    """
    return x[::-1], y[::-1]


def shift(x, y, sigma=0.015):
    """
    Shifts signal on a value, sampled from normal distribution.
    :param x: Data.
    :param y: Labels.
    :param sigma: Normal distribution std.
    :return: Shifted signal and the same labels.
    """
    sh = np.random.normal(0, sigma)
    x_ = x.copy()
    x_[:, :7] = x_[:, :7] + sh
    x_[:, 7] = x_[:, 0] ** 2
    return x_, y


def periodic_noise(x, y, max_amplitude=0.02):
    """
    Adds periodic sinusoidal signal to existing signal. Amplitude is sampled from uniform distribution.
    :param x: Data.
    :param y: Labels.
    :param max_amplitude: Maximal amplitude of added signal.
    :return: Augmented signal and existing labels.
    """
    amplitude = np.random.rand() * max_amplitude
    phase = np.random.rand() * 2 * np.pi
    frequency = 1 / 200
    time = np.arange(len(x)).reshape(-1, 1)
    periodic = amplitude * np.cos(2 * np.pi * frequency * time + phase)
    x_ = x.copy()
    x_[:, :7] = x_[:, :7] + periodic
    x_[:, 7] = x_[:, 0] ** 2
    return x_, y


def noise(x, y, sigma=0.01):
    """
    Contaminates signal with gaussian noise.
    :param x: Data.
    :param y: Labels.
    :param sigma: Normal distribution std.
    :return: Augmented signal and existing labels.
    """
    gauss = np.random.normal(0, sigma, (len(x), 1))
    x_ = x.copy()
    x_[:, :7] = x_[:, :7] + gauss
    x_[:, 7] = x_[:, 0] ** 2
    return x_, y


# Example of augmentation pipelines. It is a list of (augmentation, probability_to_apply_augmentation) tuple.
AUGS_1 = [
    (flip, 0.35),
    (shift, 0.35)
]


AUGS_2 = [
    (flip, 0.35),
    (lambda x, y: shift(x, y, sigma=0.03), 0.35)
]


# Dict for storing augmentation pipelines
AUGS = {
    1: AUGS_1,
    2: AUGS_2
}
