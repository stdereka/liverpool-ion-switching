from preprocessing.synthesis import *
from preprocessing.utils import *
import pandas as pd
import matplotlib.pyplot as plt
from seaborn import distplot
import os
import json


# Debug mode. Draws some graphics for sanity check
DEBUG = True


# Loading global variables
with open('SETTINGS.json') as settings:
    global_config = json.load(settings)


INP_DIR = global_config["RAW_DATA_DIR"]
OUT_DIR = global_config["DATA_CLEAN_DIR"]


batches_train = [slice(100000*i, 100000*(i + 5)) for i in range(0, 50, 5)]
groups_train = [0, 0, 1, 2, 3, 4, 1, 2, 4, 3]


# TODO
# Add automatic test set batch and group labeling

batches_test = [
    slice(100000*0, 100000*1),
    slice(100000*1, 100000*2),
    slice(100000*2, 100000*3),
    slice(100000*3, 100000*4),
    slice(100000*4, 100000*5),
    slice(100000*5, 100000*6),
    slice(100000*6, 100000*7),
    slice(100000*7, 100000*8),
    slice(100000*8, 100000*9),
    slice(100000*9, 100000*10),
    slice(100000*10, 100000*15),
    slice(100000*15, 100000*20),
]
groups_test = [0, 2, 4, 5, 1, 3, 4, 3, 5, 2, 5, 5]


def run_preprocessing(config: dict):
    """
    Runs preprocessing pipeline.
    :param config: Configuration dictionary (format is specified in PREPROCESSING.json).
    Must include following fields:
    RAW_DATA_TRAIN - relative path to train data,
    RAW_DATA_TEST - relative path to test data,
    CLEAN_DATA_DIR - output directory for preprocessed datasets,
    MODE - preprocessing mode. Three modes are available:
    1) DIVERSE - simulates groups 5 and 3 a of the signal using magics `group_0 + group_0 ~ group_5`
    and `group_4 + group_4 ~ group_3`. Creates large and diversified dataset,
    which helps model generalize and give better score than if trained on original data.
    2) OVERFIT - produces death dataset. Still uses `group_4 + group_4 ~ group_3` magic, but selects only highly
    correlated segments produced by combinatorial_synthesis() and causes a nightmare overfit on specific signal
    patterns. The more complex your model, the more catastrophic the consequences. GroupKFold CV will lie
    and lead you in wrong direction.
    3) REDUCED - special mode, which uses channel reduction technique. Can be used to exploit the leak.
    :return:
    """
    # Reading config
    train_path = config["RAW_DATA_TRAIN"]
    test_path = config["RAW_DATA_TEST"]
    out_dir = config["CLEAN_DATA_DIR"]
    os.makedirs(os.path.join(OUT_DIR, out_dir), exist_ok=True)
    mode = config["MODE"]
    assert mode in ["DIVERSE", "OVERFIT", "REDUCED"]

    # Loading and labeling data
    test = pd.read_csv(os.path.join(INP_DIR, test_path))
    train = pd.read_csv(os.path.join(INP_DIR, train_path))

    label_batches_and_groups(train, batches_train, groups_train)
    label_batches_and_groups(test, batches_test, groups_test)

    # Corrupted and normal noise slices on train group 2
    corrupted = slice(3_640_000, 3_840_000)
    healthy = slice(1_500_000, 1_700_000)

    # Using regression to estimate open channels mean signal values
    cleaned = train.drop(train[corrupted].index)
    signal = cleaned[cleaned.group != 3].signal.values
    channels = cleaned[cleaned.group != 3].open_channels.values

    mean_predict = predict_channel_means(signal, channels, debug=DEBUG)

    # Replace corrupted noise from train batch 7 by normal noise from another segment
    healthy_noise = train[healthy].signal.values - mean_predict[train[healthy].open_channels.values]
    fixed = mean_predict[train[corrupted].open_channels.values] + healthy_noise
    train.loc[train[corrupted].index, 'signal'] = fixed

    # Some test batches must be aligned with synthetic data
    batches_to_align = np.unique(test[test.group == 3].batch.values)

    if mode == 'DIVERSE':
        cs1 = combinatorial_synthesis(train[train.group == 0], 4, flip=False, means=mean_predict, noise_factor=2 ** 0.5)
        for sig, ch in cs1:
            train = append_dataset(train, sig, ch, 5)

        cs2 = combinatorial_synthesis(train[train.group == 4], 10, flip=False, means=mean_predict, noise_factor=1.0)
        for sig, ch in cs2:
            train = append_dataset(train, sig, ch, 3)

        # Aligning original batches of group 3 with synthetic data
        new = train.batch >= len(batches_train)
        mean_new = train[new & (train.group == 3)].signal.mean()

        for b in [4, 9]:
            train.loc[train.batch == b, 'signal'] = train[train.batch == b].signal.values - train[
                train.batch == b].signal.values.mean() + mean_new

        for b in batches_to_align:
            test.loc[test.batch == b, 'signal'] = test[test.batch == b].signal.values - test[
                test.batch == b].signal.values.mean() + mean_new

        if DEBUG:
            train.signal.plot()
            plt.show()

            test.signal.plot()
            plt.show()

            distplot(train[train.group == 3].signal.values)
            distplot(test[test.group == 3].signal.values)
            plt.show()

            distplot(train[train.group == 5].signal.values)
            distplot(test[test.group == 5].signal.values)
            plt.show()

            distplot(train[train.batch == 3].signal.values)
            distplot(train[train.batch == 7].signal.values)
            distplot(test[test.group == 2].signal.values)
            plt.show()

    if mode == 'OVERFIT':
        synthetic = [sig_ch for sig_ch in
                     combinatorial_synthesis(train[train.group == 4], 10, flip=True, means=mean_predict,
                                             noise_factor=1.0)]

        # Aligning batches
        mean_new = np.array(np.hstack([s_c[0] for s_c in synthetic])).mean()

        for b in batches_to_align:
            test.loc[test.batch == b, 'signal'] = test[test.batch == b].signal.values - test[
                test.batch == b].signal.values.mean() + mean_new

        # Select highly correlated segments
        filtered = select_corr(synthetic, synthetic[0][0], 1.5)

        for sig, ch in filtered:
            train = append_dataset(train, sig, ch, 3)

        # Remove original group 3 batches
        to_drop = (train.batch == 4) | (train.batch == 9)
        train.drop(train[to_drop].index, inplace=True)
        train.reset_index(inplace=True, drop='index')

        if DEBUG:
            train.signal.plot()
            plt.show()

            test.signal.plot()
            plt.show()

            distplot(train[train.group == 3].signal.values)
            distplot(test[test.group == 3].signal.values)
            plt.show()

            distplot(train[train.batch == 3].signal.values)
            distplot(train[train.batch == 7].signal.values)
            distplot(test[test.group == 2].signal.values)
            plt.show()

    if mode == 'REDUCED':
        train.drop(train[train.group != 4].index, inplace=True)
        train.reset_index(inplace=True, drop='index')

        test.drop(test[test.batch != 7].index, inplace=True)
        train.reset_index(inplace=True, drop='index')

        reduced_sig, residual = reduce_channels(test, train[500_000:600_000], mean_predict)

        sig = rescale_noise(train, mean_predict, 2 ** 0.5)
        train.signal = sig
        test.loc[test.batch == 7, 'signal'] = reduced_sig - reduced_sig.mean() + train.signal.mean()

        np.save(os.path.join(OUT_DIR, out_dir, "residual.npy"), residual)

        if DEBUG:
            plt.plot(np.hstack([train.signal.values, test.signal.values]))
            plt.show()

            distplot(train.signal.values)
            distplot(test.signal.values)
            plt.show()

    # Save preprocessed data
    train.to_csv(os.path.join(OUT_DIR, out_dir, "train_synthetic.csv"), index=False, float_format='%.4f')
    test.to_csv(os.path.join(OUT_DIR, out_dir, "test_synthetic.csv"), index=False, float_format='%.4f')
