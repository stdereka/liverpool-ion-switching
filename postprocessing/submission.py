import numpy as np
import pandas as pd
import json
import os


with open('SETTINGS.json') as settings:
    global_config = json.load(settings)

SUB_DIR = global_config["SUBMISSION_DIR"]
MOD_DIR = global_config["MODEL_CHECKPOINT_DIR"]
DAT_DIR = global_config["DATA_CLEAN_DIR"]


def save_submission(pred_path: str, name: str):
    """
    Writes submission in format specified by competition host.
    :param pred_path: Path to test prediction.
    :param name: Name of submission file.
    :return:
    """
    pred = np.load(os.path.join(MOD_DIR, pred_path)).argmax(axis=1)
    submission = pd.read_csv(os.path.join(SUB_DIR, "sample_submission.csv"))
    submission["open_channels"] = pred
    submission.to_csv(os.path.join(SUB_DIR, name), index=False, float_format="%.4f")


def inject_residual_prediction(sub_path: str, pred_path: str, res_path: str, name: str):
    """
    Rewrites test batch 7 of the submission with reduced channel prediction.
    :param sub_path: Path to initial submission.
    :param pred_path: Path to residual prediction.
    :param res_path: Path to residual open channels.
    :param name: Name of modified submission file.
    :return:
    """
    pred = np.load(os.path.join(MOD_DIR, pred_path)).argmax(axis=1)
    res = np.load(os.path.join(DAT_DIR, res_path))
    sub = pd.read_csv(os.path.join(SUB_DIR, sub_path))
    sub_inj = sub.copy()
    sub_inj.loc[sub_inj[700_000:800_000].index, "open_channels"] = pred + res
    sub_inj.to_csv(os.path.join(SUB_DIR, name), index=False, float_format="%.4f")
