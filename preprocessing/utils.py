import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def label_batches_and_groups(data: pd.DataFrame, batches: list, groups: list):
    """
    Adds columns 'batch' and 'group' to data.
    :param data: DataFrame with columns format specified by competition host.
    :param batches: List of batch slices.
    :param groups: List of group number, corresponding to each batch.
    :return:
    """
    data.loc[:, 'batch'] = np.empty(len(data), dtype=np.int)
    data.loc[:, 'group'] = np.empty(len(data), dtype=np.int)

    for i in range(len(batches)):
        b = batches[i]
        g = groups[i]
        data.loc[b, 'batch'] = i
        data.loc[b, 'group'] = g


def get_stats(sig, chan):
    """
    Computes mean and std signal statistics for each number of opened channels in the data.
    :param sig: Numpy array of signal.
    :param chan: Numpy array of open channels.
    :return: Four numpy arrays corresponding channels, counts of each channel, means and stds.
    """
    channels, counts = np.unique(chan, return_counts=True)
    means = []
    stds = []
    for ch in channels:
        sig_ = sig[chan == ch]
        means.append(sig_.mean())
        stds.append(sig_.std())
    return channels, counts, np.array(means), np.array(stds)


def predict_channel_means(signal: np.ndarray, channels: np.ndarray, max_channels=10, debug=False):
    """
    Computes linear regression predictions for open channels mean signal values.
    :param signal: Numpy array of signal.
    :param channels: Numpy array of open channels.
    :param max_channels: Max number of opened channels to predict.
    :param debug: Draw predicted means on a graph.
    :return: mean_predict - numpy array of predicted mean values.
    """
    channel_list, n_list, mean_list, std_list = get_stats(signal, channels)

    stderr_list = std_list / np.sqrt(n_list)
    plt.show()

    w = 1 / stderr_list
    channel_list = channel_list.reshape(-1, 1)
    linreg_m = LinearRegression()
    linreg_m.fit(channel_list, mean_list, sample_weight=w)

    mean_predict = linreg_m.predict(np.arange(0, max_channels + 1).reshape(-1, 1))

    if debug:
        x = np.linspace(-0.5, 10, 5)
        y = linreg_m.predict(x.reshape(-1, 1))
        plt.plot(x, y, label="regression")
        plt.plot(channel_list, mean_list, ".", markersize=8, label="original")
        plt.legend()
        plt.show()

        print("mean:", mean_predict)
        print("std:", std_list)
        print("stderr:", stderr_list)

    return mean_predict
