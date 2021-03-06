import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix


def print_model_quality_report(pred_path: str, ground_path: str):
    """
    Prints various metrics for the model.
    :param pred_path: Path to model's train OOF predictions.
    :param ground_path: Path to train data .csv file. Should be generated by preprocessing pipeline.
    :return:
    """
    predictions = np.load(pred_path).argmax(axis=1)
    groundtruth = pd.read_csv(ground_path).open_channels.values
    groups = pd.read_csv(ground_path).group.values

    print("Macro F1 score, F1 scores and confusion matrix per group:")
    for group in range(6):
        pred = predictions[groups == group]
        true = groundtruth[groups == group]
        print(f"Group {group} macro F1 score, F1 scores and confusion matrix:")
        print(f1_score(true, pred, average='macro'))
        print(f1_score(true, pred, average=None))
        print(confusion_matrix(true, pred, normalize='true').round(3))
        print()

    print("Batch 5 macro F1 score, F1 scores and confusion matrix:")
    pred = predictions[2_000_000:2_500_000]
    true = groundtruth[2_000_000:2_500_000]
    print(f1_score(true, pred, average='macro'))
    print(f1_score(true, pred, average=None))
    print(confusion_matrix(true, pred, normalize='true').round(3))
    print()

    print("Batch 9 macro F1 score, F1 scores and confusion matrix:")
    pred = predictions[4_500_000:5_000_000]
    true = groundtruth[4_500_000:5_000_000]
    print(f1_score(true, pred, average='macro'))
    print(f1_score(true, pred, average=None))
    print(confusion_matrix(true, pred, normalize='true').round(3))
    print()

    print("Overall OOF macro F1 score, F1 scores and confusion matrix:")
    print(f1_score(groundtruth[:5_000_000], predictions[:5_000_000], average='macro'))
    print(f1_score(groundtruth[:5_000_000], predictions[:5_000_000], average=None))
    print(confusion_matrix(groundtruth[:5_000_000], predictions[:5_000_000], normalize='true').round(3))
    print()
