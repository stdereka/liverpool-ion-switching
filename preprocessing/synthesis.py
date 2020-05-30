import numpy as np
import pandas as pd
from itertools import combinations
from preprocessing.utils import get_stats


def compose_old(data_1: pd.DataFrame, data_2: pd.DataFrame, mean_0: float, noise_factor: float):
    """
    Computes superposition of two signals. Old version.
    :param data_1: First data segment with columns format specified by competition host.
    :param data_2: Second data segment with columns format specified by competition host.
    :param mean_0: Mean signal value of 0 open channels.
    :param noise_factor: Noise scaling factor.
    :return: comp - numpy array of signals, comp_label - numpy array of opened channels.
    """
    _, means_1, stds_1 = get_stats(data_1)
    _, means_2, stds_2 = get_stats(data_2)

    sh_1 = means_1[1:] - means_1[:-1]
    sh_2 = means_2[1:] - means_2[:-1]
    shift = np.median(np.hstack((sh_1, sh_2)))

    ch_1 = data_1.open_channels.values
    ch_2 = data_2.open_channels.values
    comp_label = ch_1 + ch_2

    noise_1 = data_1.signal.values - means_1[ch_1]
    noise_2 = data_2.signal.values - means_2[ch_2]
    noise = (noise_1 + noise_2) / noise_factor

    comp = mean_0 + shift * comp_label + noise
    return comp, comp_label


def compose(data_1: pd.DataFrame, data_2: pd.DataFrame, means: np.ndarray, noise_factor: float):
    """
    Computes superposition of two signals. Actual version. The motivation behind this method is explained here:
    https://www.kaggle.com/stdereka/2nd-place-solution-preprocessing-tricks
    :param data_1: First data segment with columns format specified by competition host.
    :param data_2: Second data segment with columns format specified by competition host.
    :param means: Numpy array of shape (max_open_channels,). Contains mean signal values for all channels.
    :param noise_factor: Noise scaling factor.
    :return: comp - numpy array of signals, comp_label - numpy array of opened channels.
    """
    ch_1 = data_1.open_channels.values
    ch_2 = data_2.open_channels.values
    comp_label = ch_1 + ch_2

    noise_1 = data_1.signal.values - means[ch_1]
    noise_2 = data_2.signal.values - means[ch_2]
    noise = (noise_1 + noise_2) / noise_factor

    comp = means[comp_label] + noise
    return comp, comp_label


def combinatorial_synthesis(data: pd.DataFrame, n: int, flip: bool, **params):
    """
    Uses compose() to generate \binom{n}{2}(1 + flip) segments of length len(data) // n of synthetic data.
    :param data: Data segment with columns format specified by competition host.
    :param n: Number of parts to split the data into.
    :param flip: Pass flipped segments into compose.
    :param params: Parameters for compose().
    :return: sig - numpy array of signals, ch - numpy array of opened channels \binom{n}{2}(1 + flip) times
    """
    assert len(data) % n == 0
    l_s = len(data) // n
    comb = combinations(list(range(n)), 2)
    for i, j in comb:
        sig, ch = compose(data[i*l_s:(i+1)*l_s], data[j*l_s:(j+1)*l_s], **params)
        yield sig, ch
        if flip:
            sig, ch = compose(data[i * l_s:(i + 1) * l_s], data[j * l_s:(j + 1) * l_s][::-1], **params)
            yield sig, ch


def append_dataset(data: pd.DataFrame, signal: np.ndarray, channels: np.ndarray, group: int):
    """
    Appends existing dataset with new data.
    :param data: Data segment with columns format specified by competition host. Must include 'batch' and 'group'.
    :param signal: Numpy array with new data signal.
    :param channels: Numpy array with new data open channels.
    :param group: Group of new segment.
    :return:
    """
    t_0 = data.time.values[-1]
    b = data.batch.values[-1]
    tau = 0.0001
    time = np.arange(t_0 + tau, t_0 + tau * (len(signal) + 1), tau)
    new = pd.DataFrame()
    new['time'] = time
    new['signal'] = signal
    new['open_channels'] = channels
    new['batch'] = b + 1
    new['group'] = group
    return pd.concat([data, new], ignore_index=True, axis=0)


def append_non_corr(l1: list, l2: list, thresh: float):
    """
    Appends list l1 of segments with samples from l2 which don't correlate with any of segments in l1.
    :param l1: First list of (signal, channels) segments.
    :param l2: Second list of (signal, channels) segments.
    :param thresh: Minimal MAE between two segments to consider them non-correlated.
    :return:
    """
    for sig in l2:
        corrs = [np.mean(np.abs(sig[0] - sig_[0])) >= thresh for sig_ in l1]
        if all(corrs):
            l1.append(sig)


def select_corr(l: list, sig: np.ndarray, thresh: float):
    """
    Filters data segments from l which are correlated with sig.
    :param l: List of (signal, channels) segments.
    :param sig: Signal to compare with.
    :param thresh: Minimal MAE between two segments to consider them non-correlated.
    :return: res - list of filtered segments.
    """
    res = []
    for sig_, ch_ in l:
        if np.mean(np.abs(sig - sig_)) < thresh:
            res.append((sig_, ch_))
    return res


def rescale_noise(data: pd.DataFrame, means: np.ndarray, scale_factor: float):
    """
    Extracts noise from data and rescales it.
    :param data: Data segment with columns format specified by competition host.
    :param means: Numpy array of shape (max_open_channels,). Contains mean signal values for all channels.
    :param scale_factor: Noise scaling factor.
    :return: sig_ - modified signal from data.
    """
    sig = data.signal.values
    ch = data.open_channels.values
    noise = sig - means[ch]
    noise *= scale_factor
    sig_ = noise + means[ch]
    return sig_


def reduce_channels(data: pd.DataFrame, res: pd.DataFrame, means: np.ndarray):
    """
    Subtracts open channel values of another segment from the data. Can be used to exploit data leak.
    :param data: Data segment with columns format specified by competition host.
    :param res: Data segment, which channels will be subtracted.
    :param means: Numpy array of shape (max_open_channels,). Contains mean signal values for all channels.
    :return: reduced_sig, residual - modified signal and array of subtracted channels.
    """
    residual = res.open_channels.values
    reduced_sig = data.signal.values - (means - means[0])[residual]
    return reduced_sig, residual
