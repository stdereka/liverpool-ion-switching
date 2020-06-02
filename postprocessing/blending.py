import json
import numpy as np
import os


with open('SETTINGS.json') as settings:
    global_config = json.load(settings)

MOD_DIR = global_config["MODEL_CHECKPOINT_DIR"]


def blend_predictions(config: dict):
    """
    Blends OOF and test predictions of given models.
    :param config: Configuration dictionary (format is specified in ./config/BLENDING.json).
    Must include following fields:
    TEST_PREDICTIONS_PATHS - path to test model predictions,
    OOF_PREDICTIONS_PATHS - path to OOF model predictions,
    OUT_DIR - directory for storing blended probabilities.
    :return:
    """
    preds = config["TEST_PREDICTIONS_PATHS"]
    oofs = config["OOF_PREDICTIONS_PATHS"]
    out_dir = config["OUT_DIR"]

    os.makedirs(os.path.join(MOD_DIR, out_dir), exist_ok=True)

    test_probas = []
    for pred in preds:
        test_probas.append(np.load(os.path.join(MOD_DIR, pred)))

    test_proba = sum(test_probas) / len(test_probas)

    train_probas = []
    for pred in oofs:
        train_probas.append(np.load(os.path.join(MOD_DIR, pred)))

    train_proba = sum(train_probas) / len(train_probas)

    np.save(os.path.join(MOD_DIR, out_dir, "train_proba.npy"), train_proba)
    np.save(os.path.join(MOD_DIR, out_dir, "test_proba.npy"), test_proba)
