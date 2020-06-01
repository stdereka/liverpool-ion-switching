import tensorflow as tf
import pandas as pd
import numpy as np
import random
import os
import json
from tensorflow.keras.callbacks import Callback, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras import losses
import gc
import joblib
from model.nn import get_model
from model.augs import AUGS
from model.datagen import DataGenerator
from model.utils import StandardScaler, MinMaxScaler, ColumnDropper, read_input

from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 500)

with open('SETTINGS.json') as settings:
    global_config = json.load(settings)

MOD_DIR = global_config["MODEL_CHECKPOINT_DIR"]
SUB_DIR = global_config["SUBMISSION_DIR"]
DAT_DIR = global_config["DATA_CLEAN_DIR"]

CD = ColumnDropper(columns=["time", "batch", "group", "oversample"])

SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)


def batching(df, batch_size):
    df['group'] = df.groupby(df.index // batch_size, sort=False)['signal'].agg(['ngroup']).values
    df['group'] = df['group'].astype(np.uint16)
    return df


def add_rfc_proba(df, proba_path):
    proba = np.load(proba_path)

    for i in range(proba.shape[1]):
        df[f"proba_rfc_{i}"] = proba[:, i]

    return df


def lag_with_pct_change(df, windows):
    for window in windows:
        df['signal_shift_pos_' + str(window)] = df.groupby('group')['signal'].shift(window).fillna(0)
        df['signal_shift_neg_' + str(window)] = df.groupby('group')['signal'].shift(-1 * window).fillna(0)
    return df


def run_feat_engineering(df, batch_size):
    df = batching(df, batch_size=batch_size)
    df = lag_with_pct_change(df, [1, 2, 3])
    df['signal_2'] = df['signal'] ** 2
    return df


def feature_selection(df):
    features = [col for col in df.columns if col not in ['index', 'group', 'open_channels', 'time', 'oversample']]
    df = df.replace([np.inf, -np.inf], np.nan)
    for feature in features:
        feature_mean = df[feature].mean()
        df[feature] = df[feature].fillna(feature_mean)
    return df, features


# function that decrease the learning as epochs increase (i also change this part of the code)
def lr_schedule(epoch, lr):
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


class MacroF1(Callback):
    def __init__(self, model, inputs, targets):
        self.model = model
        self.inputs = inputs
        self.targets = np.argmax(targets, axis=2).reshape(-1)

    def on_epoch_end(self, epoch, logs):
        pred = np.argmax(self.model.predict(self.inputs), axis=2).reshape(-1)
        score = f1_score(self.targets, pred, average='macro')
        print(f'F1 Macro Score: {score:.5f}')


def run_cv_model_by_batch(train, splits, feats, nn_epochs, nn_batch_size, seed, lr, save_dir, version, n_classes):
    seed_everything(seed)
    K.clear_session()
    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config)
    tf.compat.v1.keras.backend.set_session(sess)
    oof_ = np.zeros((len(train), n_classes))
    target = ['open_channels']
    group = train['group']
    kf = GroupKFold(n_splits=splits)
    splits = [x for x in kf.split(train, train[target], group)]

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

    for n_fold, (tr_idx, val_idx, val_orig_idx) in enumerate(new_splits[0:], start=0):
        train_x, train_y = train[tr_idx], train_tr[tr_idx]
        valid_x, valid_y = train[val_idx], train_tr[val_idx]
        print(f'Our training dataset shape is {train_x.shape}')
        print(f'Our validation dataset shape is {valid_x.shape}')

        train_gen = DataGenerator(train_x, train_y, batch_size=nn_batch_size, shuffle=True, mode='train', augs=AUGS)
        val_gen = DataGenerator(valid_x, valid_y, batch_size=nn_batch_size, shuffle=False, mode='val', augs=None)

        e_s = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=25,
            verbose=1,
            restore_best_weights=True,
        )

        gc.collect()
        shape_ = (None, train_x.shape[2])

        opt = Adam(lr=lr)
        loss = losses.CategoricalCrossentropy()
        model = get_model(version=version, shape=shape_, n_classes=n_classes, loss=loss, opt=opt)

        cb_lr_schedule = LearningRateScheduler(lambda x: lr_schedule(x, lr))
        model.fit_generator(generator=train_gen,
                            epochs=nn_epochs,
                            callbacks=[cb_lr_schedule, MacroF1(model, valid_x, valid_y), e_s],
                            verbose=2,
                            validation_data=val_gen)
        model.save(os.path.join(save_dir, f"wavenet_fold_{n_fold}.h5"))

        preds_f = model.predict(valid_x)
        f1_score_ = f1_score(np.argmax(valid_y, axis=2).reshape(-1),
                             np.argmax(preds_f, axis=2).reshape(-1), average='macro')
        print(f'Training fold {n_fold} completed. macro f1 score : {f1_score_ :1.5f}')
        preds_f = preds_f.reshape(-1, preds_f.shape[-1])
        oof_[val_orig_idx, :] += preds_f

    np.save('train_wavenet_proba.npy', oof_)
    f1_score_ = f1_score(np.argmax(train_tr, axis=2).reshape(-1), np.argmax(oof_, axis=1), average='macro')
    print(f'Training completed. oof macro f1 score : {f1_score_:1.5f}')


def train_wavenet(config):
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

    os.makedirs(os.path.join(MOD_DIR, out_dir), exist_ok=True)

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
    train = add_rfc_proba(train, os.path.join(MOD_DIR, rfc_proba_dir, "train_rfc_proba.npy"))
    train, features = feature_selection(train)
    print(train)
    print('Feature Engineering Completed...')

    print('Features:', *list(features), sep='\n')

    print(f'Training Wavenet model with {n_splits} folds of GroupKFold Started...')
    run_cv_model_by_batch(train, n_splits, features, epochs, nnbatchsize, seed, lr,
                          os.path.join(MOD_DIR, out_dir), version, n_classes)

    print('Training completed...')


def predict_wavenet():
    pass
