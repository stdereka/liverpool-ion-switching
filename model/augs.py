import numpy as np


def flip(x, y):
    return x[::-1], y[::-1]


def shift(x, y, sigma=0.015):
    sh = np.random.normal(0, sigma)
    x_ = x.copy()
    x_[:, :7] = x_[:, :7] + sh
    x_[:, 7] = x_[:, 0] ** 2
    return x_, y


def periodic_noise(x, y, max_amplitude=0.02):
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
    gauss = np.random.normal(0, sigma, (len(x), 1))
    x_ = x.copy()
    x_[:, :7] = x_[:, :7] + gauss
    x_[:, 7] = x_[:, 0] ** 2
    return x_, y


AUGS = [
    (flip, 0.35),
    (shift, 0.35)
]
