import numpy as np
import gc
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from model.utils import ShiftedFeatureMaker, ColumnDropper, add_category, read_input, save_submission
import os
import joblib


# Global variables
with open('SETTINGS.json') as settings:
    global_config = json.load(settings)

MOD_DIR = global_config["MODEL_CHECKPOINT_DIR"]
SUB_DIR = global_config["SUBMISSION_DIR"]
DAT_DIR = global_config["DATA_CLEAN_DIR"]

FM = ShiftedFeatureMaker(periods=list(range(1, 20)), add_minus=True, fill_value=0)
CD = ColumnDropper(columns=["open_channels", "time", "batch", "group", "oversample"])


def train_rfc(config):
    """
    Runs RFC training pipeline. Trains RFC model and stores its weights and OOF predictions on disk.
    :param config: Configuration dictionary (format is specified in ./config/RFC.json).
    Must include following fields:
    CLEAN_TRAIN_DATA_PATH - path to preprocessed train data,
    CHECKPOINT_DIR - directory for storing model and prediction,
    N_SPLITS - number of CV splits,
    N_CLASSES - number of classes to predict,
    STRAT_INTERVAL - stratification interval length. CV fold data will be stratified by intervals.
    RFC_PARAMS - parameters for sklearn.ensemble.RandomForestClassifier model.
    :return:
    """
    # Load config
    train_path = config["CLEAN_TRAIN_DATA_PATH"]
    out_dir = config["CHECKPOINT_DIR"]
    params = config["RFC_PARAMS"]
    n_splits = config["N_SPLITS"]
    n_classes = config["N_CLASSES"]
    strat_int = config["STRAT_INTERVAL"]
    os.makedirs(os.path.join(MOD_DIR, out_dir), exist_ok=True)

    train = read_input(os.path.join(DAT_DIR, train_path))
    train = add_category(train)
    y = train.open_channels.values

    train = FM.transform(train)
    print("Train size: {}".format(len(train)))

    train = CD.transform(train)
    gc.collect()

    # Validation setup
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42).split(
        train, train.index // strat_int
    )

    # Array for storing OOF predictions
    oof = np.zeros((len(train), n_classes), np.float32)

    # Train on folds and make OOF predictions
    folds = [x for x in cv]
    for n, fold in enumerate(folds):
        train_idx, val_idx = fold
        x_train, y_train = train.values[train_idx], y[train_idx]
        x_val, y_val = train.values[val_idx], y[val_idx]
        model = RandomForestClassifier(**params)
        model.fit(x_train, y_train)
        del x_train, y_train
        gc.collect()
        pred = model.predict_proba(x_val)
        score = f1_score(y_val, pred.argmax(axis=1), average='macro')
        print('Fold {}, macro F1 score: {}'.format(n, score))
        oof[val_idx] = pred
        joblib.dump(model, os.path.join(MOD_DIR, out_dir, f"rfc_fold_{n}.joblib"))

    np.save(os.path.join(MOD_DIR, out_dir, "train_rfc_proba.npy"), oof)
    print('OOF macro F1 score:', f1_score(y, oof.argmax(axis=1), average='macro'))


def predict_rfc(config):
    """
    Runs RFC inference pipeline. Uses serialized copy of the model, created by train_rfc().
    :param config: Configuration dictionary (format is specified in ./config/RFC.json).
    Must include following fields:
    CLEAN_TEST_DATA_PATH - path to preprocessed test data,
    CHECKPOINT_DIR - directory for storing model and prediction,
    N_SPLITS - number of CV splits,
    N_CLASSES - number of classes to predict.
    :return:
    """
    test_path = config["CLEAN_TEST_DATA_PATH"]
    out_dir = config["CHECKPOINT_DIR"]
    n_splits = config["N_SPLITS"]
    n_classes = config["N_CLASSES"]

    test = read_input(os.path.join(DAT_DIR, test_path))
    test = add_category(test)

    test = FM.transform(test)
    print("Test size: {}".format(len(test)))

    test = CD.transform(test)
    test = test.values
    gc.collect()

    # Test prediction is a mean of N_SPLITS models trained on different folds
    test_pred = np.zeros((len(test), n_classes), np.float32)
    for n in range(n_splits):
        model = joblib.load(os.path.join(MOD_DIR, out_dir, f"rfc_fold_{n}.joblib"))
        test_pred += model.predict_proba(test)
    test_pred /= n_splits

    np.save(os.path.join(MOD_DIR, out_dir, "test_rfc_proba.npy"), test_pred)
