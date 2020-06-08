import tensorflow as tf
import pandas as pd
import numpy as np
import random
import os
import json
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras import losses, models
import gc
import joblib
from model.nn import get_model
from model.augs import AUGS
from model.datagen import DataGenerator
from model.utils import StandardScaler, MinMaxScaler, ColumnDropper, read_input, MacroF1
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 500)


# Global variables
with open('SETTINGS.json') as settings:
    global_config = json.load(settings)

MOD_DIR = global_config["MODEL_CHECKPOINT_DIR"]
SUB_DIR = global_config["SUBMISSION_DIR"]
DAT_DIR = global_config["DATA_CLEAN_DIR"]

CD = ColumnDropper(columns=["time", "batch", "group", "oversample"])


# All scalers available at this moment.
SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler
}


def batching(df: pd.DataFrame, batch_size: int):
    """
    Splits signal into sequences of given length.
    :param df: Initial DataFrame.
    :param batch_size: Length of sequence to split into.
    :return: Extended DataFrame with 'group' column.
    """
    df['group'] = df.groupby(df.index // batch_size, sort=False)['signal'].agg(['ngroup']).values
    df['group'] = df['group'].astype(np.uint16)
    return df


def seed_everything(seed: int):
    """
    Sets random seeds for used packages.
    :param seed: Random seed.
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)


def add_rfc_proba(df: pd.DataFrame, proba_path: str):
    """
    Extends DataFrame with OOF RFC probabilities.
    :param df: Initial DataFrame.
    :param proba_path: Path to OOF RFC probabilities directory.
    :return: Extended DataFrame.
    """
    proba = np.load(proba_path)

    for i in range(proba.shape[1]):
        df[f"proba_rfc_{i}"] = proba[:, i]

    return df


def lag_with_pct_change(df: pd.DataFrame, windows: list):
    """
    Computes lag and lead signal features.
    :param df: Initial DataFrame.
    :param windows: List of lag and lead orders to compute.
    :return: DataFrame, extended with lag and lead features.
    """
    for window in windows:
        df['signal_shift_pos_' + str(window)] = df.groupby('group')['signal'].shift(window).fillna(0)
        df['signal_shift_neg_' + str(window)] = df.groupby('group')['signal'].shift(-1 * window).fillna(0)
    return df


def run_feat_engineering(df: pd.DataFrame, batch_size: int):
    """
    Generates features for wavenet.
    :param df: Initial DataFrame.
    :param batch_size: Size of signal sequence.
    :return: Transformed DataFrame
    """
    df = batching(df, batch_size=batch_size)
    df = lag_with_pct_change(df, [1, 2, 3])
    df['signal_2'] = df['signal'] ** 2
    return df


def feature_selection(df: pd.DataFrame):
    """
    Selects features and fills all the gaps with mean values.
    :param df: DataFrame.
    :return: Transformed df and list of feature names.
    """
    features = [col for col in df.columns if col not in ['index', 'group', 'open_channels', 'time', 'oversample']]
    df = df.replace([np.inf, -np.inf], np.nan)
    for feature in features:
        feature_mean = df[feature].mean()
        df[feature] = df[feature].fillna(feature_mean)
    return df, features


def lr_schedule(epoch: int, lr: float):
    """
    Scheduler function for keras callback.
    :param epoch: Epoch number.
    :param lr: Initial learning rate.
    :return: Updated LR.
    """
    if epoch < 40:
        lr_ = lr
    elif epoch < 60:
        lr_ = lr / 3
    elif epoch < 70:
        lr_ = lr / 5
    elif epoch < 80:
        lr_ = lr / 7
    elif epoch < 90:
        lr_ = lr / 9
    elif epoch < 100:
        lr_ = lr / 11
    elif epoch < 110:
        lr_ = lr / 13
    else:
        lr_ = lr / 100
    return lr_


def run_train_cycle(train: pd.DataFrame, splits: int, feats: list,
                    nn_epochs: int, nn_batch_size: int, seed: int,
                    lr: float, save_dir: str, version: int, n_classes: int, augs: dict):
    """
    Wavenet training cycle. Runs GroupKFold crossvalidation. Saves model for each fold.
    :param train: DataFrame with training data.
    :param splits: Number of folds in CV.
    :param feats: List of features for training.
    :param nn_epochs: Number of epochs to train.
    :param nn_batch_size: Batch size.
    :param seed: Random seed.
    :param lr: Learning rate.
    :param save_dir: Directory for storing models and OOF predictions.
    :param version: Model version. Specified in nn.py.
    :param n_classes: Number of classes.
    :return:
    """
    seed_everything(seed)
    K.clear_session()
    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config)
    tf.compat.v1.keras.backend.set_session(sess)
    oof_ = np.zeros((len(train), n_classes))
    target = ['open_channels']
    group = train['group']
    # Setup GroupKFold validation
    kf = GroupKFold(n_splits=splits)
    splits = [x for x in kf.split(train, train[target], group)]

    # Find batches corresponding to validation splits
    new_splits = []
    for sp in splits:
        new_split = []
        tr_idx = np.unique(group[sp[0]])
        new_split.append(tr_idx)
        new_split.append(np.unique(group[sp[1]]))
        new_split.append(sp[1])
        new_splits.append(new_split)

    tr = pd.concat([pd.get_dummies(train.open_channels), train[['group']]], axis=1)

    tr.columns = ['target_' + str(i) for i in range(n_classes)] + ['group']
    target_cols = ['target_' + str(i) for i in range(n_classes)]
    train_tr = np.array(list(tr.groupby('group').apply(lambda x: x[target_cols].values))).astype(np.float32)
    train = np.array(list(train.groupby('group').apply(lambda x: x[feats].values)))

    # Train <splits> models
    for n_fold, (tr_idx, val_idx, val_orig_idx) in enumerate(new_splits[0:], start=0):
        train_x, train_y = train[tr_idx], train_tr[tr_idx]
        valid_x, valid_y = train[val_idx], train_tr[val_idx]
        print(f'Our training dataset shape is {train_x.shape}')
        print(f'Our validation dataset shape is {valid_x.shape}')

        # Data generators
        train_gen = DataGenerator(train_x, train_y, batch_size=nn_batch_size, shuffle=True, mode='train', augs=augs)
        val_gen = DataGenerator(valid_x, valid_y, batch_size=nn_batch_size, shuffle=False, mode='val', augs=None)

        # Early stopping configuration
        e_s = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=25,
            verbose=1,
            restore_best_weights=True,
        )

        gc.collect()
        shape_ = (None, train_x.shape[2])

        # Model
        opt = Adam(lr=lr)
        loss = losses.CategoricalCrossentropy()
        model = get_model(version=version, shape=shape_, n_classes=n_classes, loss=loss, opt=opt)

        # Learning scheduler is used
        cb_lr_schedule = LearningRateScheduler(lambda x: lr_schedule(x, lr))

        model.fit_generator(
            generator=train_gen,
            epochs=nn_epochs,
            callbacks=[cb_lr_schedule, MacroF1(model, valid_x, valid_y), e_s],
            verbose=2,
            validation_data=val_gen
        )

        # Save weights to disc
        model.save(os.path.join(save_dir, f"wavenet_fold_{n_fold}.h5"))

        # Write OOF predictions and compute F1 score for the fold
        preds_f = model.predict(valid_x)
        f1_score_ = f1_score(np.argmax(valid_y, axis=2).reshape(-1),
                             np.argmax(preds_f, axis=2).reshape(-1), average='macro')
        print(f'Training fold {n_fold} completed. macro f1 score : {f1_score_ :1.5f}')
        preds_f = preds_f.reshape(-1, preds_f.shape[-1])
        oof_[val_orig_idx, :] += preds_f

    # Save OOF array and compute Overall OOF score
    np.save(os.path.join(save_dir, "train_wavenet_proba.npy"), oof_)
    f1_score_ = f1_score(np.argmax(train_tr, axis=2).reshape(-1), np.argmax(oof_, axis=1), average='macro')
    print(f'Training completed. oof macro f1 score : {f1_score_:1.5f}')


def train_wavenet(config: dict):
    """
    Wavenet training pipeline. Prepares data and runs training cycle.
    :param config: Configuration dictionary (format is specified in ./config/WAVENET.json).
    Must include following fields:
    CLEAN_TRAIN_DATA_PATH - relative path to preprocessed train data,
    CHECKPOINT_DIR - directory for storing model and prediction,
    SCALER - scaler to use. "standard" and "minmax" scalers are available,
    N_SPLITS - number of CV splits,
    N_CLASSES - number of classes to predict,
    EPOCHS - number of training epochs,
    NNBATCHSIZE - batch size,
    GROUP_BATCH_SIZE - length of sequence accepted by wavenet.
    Wavenet input layer shape is (NNBATCHSIZE, GROUP_BATCH_SIZE, number_of_features),
    SEED - random seed,
    LR - learning rate,
    MODEL_VERSION - version of wavenet to use. Read documentation for nn.py to know more,
    RFC_PROBA_DIR - storing directory of RFC model. If None, wavenet will be trained without RFC probabilities,
    AUGS_VERSION - augmentation pipeline version.
    :return:
    """
    # Read config
    data_path = config["CLEAN_TRAIN_DATA_PATH"]
    out_dir = config["CHECKPOINT_DIR"]
    scaler = config["SCALER"]
    n_splits = config["N_SPLITS"]
    n_classes = config["N_CLASSES"]
    epochs = config["EPOCHS"]
    nnbatchsize = config["NNBATCHSIZE"]
    group_batch_size = config["GROUP_BATCH_SIZE"]
    seed = config["SEED"]
    lr = config["LR"]
    version = config["MODEL_VERSION"]
    rfc_proba_dir = config["RFC_PROBA_DIR"]
    augs_version = config["AUGS_VERSION"]

    os.makedirs(os.path.join(MOD_DIR, out_dir), exist_ok=True)

    # Prepare data
    print('Reading Data Started...')
    train = read_input(os.path.join(DAT_DIR, data_path))

    train = CD.transform(train)

    scaler = SCALERS[scaler]()
    scaler.fit(train, None)
    train = scaler.transform(train, None)
    joblib.dump(scaler, os.path.join(MOD_DIR, out_dir, "scaler.joblib"))

    print('Reading and Normalizing Data Completed')

    print('Creating Features')
    print('Feature Engineering Started...')
    train = run_feat_engineering(train, batch_size=group_batch_size)
    if rfc_proba_dir:
        train = add_rfc_proba(train, os.path.join(MOD_DIR, rfc_proba_dir, "train_rfc_proba.npy"))
    train, features = feature_selection(train)
    print('Feature Engineering Completed...')

    print('Features:', *list(features), sep='\n')

    # Launch training process
    print(f'Training Wavenet model with {n_splits} folds of GroupKFold Started...')
    run_train_cycle(train, n_splits, features, epochs, nnbatchsize, seed, lr,
                    os.path.join(MOD_DIR, out_dir), version, n_classes, AUGS[augs_version])

    print('Training completed...')


def predict_wavenet(config: dict):
    """
    Runs wavenet inference pipeline. Uses serialized copy of the model, created by train_wavenet().
    :param config: Configuration dictionary (format is specified in ./config/WAVENET.json).
    Must include following fields:
    CLEAN_TEST_DATA_PATH - relative path to preprocessed test data,
    CHECKPOINT_DIR - directory with stored model,
    N_SPLITS - number of CV splits,
    N_CLASSES - number of classes to predict,
    GROUP_BATCH_SIZE - length of sequence accepted by wavenet.
    Wavenet input layer shape is (NNBATCHSIZE, GROUP_BATCH_SIZE, number_of_features),
    RFC_PROBA_DIR - storing directory of RFC model. Should be None, of no RFC probabilities were used in training.
    :return:
    """
    # Read config
    data_path = config["CLEAN_TEST_DATA_PATH"]
    out_dir = config["CHECKPOINT_DIR"]
    n_splits = config["N_SPLITS"]
    n_classes = config["N_CLASSES"]
    group_batch_size = config["GROUP_BATCH_SIZE"]
    rfc_proba_dir = config["RFC_PROBA_DIR"]

    # Prepare test data
    test = read_input(os.path.join(DAT_DIR, data_path))
    test = CD.transform(test)
    scaler = joblib.load(os.path.join(MOD_DIR, out_dir, "scaler.joblib"))
    test = scaler.transform(test, None)
    test = run_feat_engineering(test, batch_size=group_batch_size)
    if rfc_proba_dir:
        test = add_rfc_proba(test, os.path.join(MOD_DIR, rfc_proba_dir, "test_rfc_proba.npy"))
    test, features = feature_selection(test)
    test = np.array(list(test.groupby('group').apply(lambda x: x[features].values)))

    # Test prediction is a mean of N_SPLITS models trained on different folds
    test_pred = np.zeros((test.shape[0]*test.shape[1], n_classes))
    for n in range(n_splits):
        model = models.load_model(os.path.join(MOD_DIR, out_dir, f"wavenet_fold_{n}.h5"))
        te_preds = model.predict(test)
        del model
        gc.collect()
        te_preds = te_preds.reshape(-1, te_preds.shape[-1])
        test_pred += te_preds

    test_pred /= n_splits

    # Save array with probabilities
    np.save(os.path.join(MOD_DIR, out_dir, "test_wavenet_proba.npy"), test_pred)
