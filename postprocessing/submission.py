import numpy as np
import pandas as pd
import json
import os


with open('SETTINGS.json') as settings:
    global_config = json.load(settings)

SUB_DIR = global_config["SUBMISSION_DIR"]
MOD_DIR = global_config["MODEL_CHECKPOINT_DIR"]


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
