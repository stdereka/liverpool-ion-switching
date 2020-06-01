import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Scaler:
    def fit(self, x, y=None):
        """
        Computes Scaler parameters and updates corresponding fields.
        :param x: Data.
        :param y: Labels.
        :return:
        """
        raise NotImplementedError

    def transform(self, x, y=None):
        """
        Performs transformation on new data.
        :param x: Data. Must include 'signal' column.
        :param y: Labels.
        :return: Transformed x.
        """
        raise NotImplementedError


# Data normalizer
class StandardScaler(Scaler):
    def __init__(self):
        self.mean = None
        self.sigma = None

    def fit(self, x: pd.DataFrame, y=None):
        self.mean = x.signal.mean()
        self.sigma = x.signal.std()
        return self

    def transform(self, x: pd.DataFrame, y=None):
        x['signal'] = (x.signal - self.mean) / self.sigma
        return x


# Minmax scaling. Transformed signal is in diapason [0, 1]
class MinMaxScaler(Scaler):
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, x, y):
        self.min = x.signal.min()
        self.max = x.signal.max()
        return self

    def transform(self, x, y):
        x['signal'] = (x.signal - self.min) / (self.max - self.min)
        return x


# Appends dataframe with lag and lead features
class ShiftedFeatureMaker(BaseEstimator, TransformerMixin):
    def __init__(self, periods=[1], column="signal", add_minus=False, fill_value=None, copy=True):
        self.periods = periods
        self.column = column
        self.add_minus = add_minus
        self.fill_value = fill_value
        self.copy = copy

    def fit(self, x, y):
        """Mock method"""
        return self

    def transform(self, x: pd.DataFrame, y=None):
        """
        Creates dataframe with lag and lead columns.
        :param x: Data. Must include 'signal' column.
        :param y: Labels.
        :return: x_transformed - pd.DataFrame with lag and lead columns.
        """
        periods = np.asarray(self.periods, dtype=np.int32)

        if self.add_minus:
            periods = np.append(periods, -periods)

        x_transformed = x.copy() if self.copy else x

        for p in periods:
            x_transformed[f"{self.column}_shifted_{p}"] = x_transformed[self.column].shift(
                periods=p, fill_value=self.fill_value
            )

        return x_transformed


# Drops columns which are marked as useless.
class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y):
        """Mock method"""
        return self

    def transform(self, x: pd.DataFrame, y=None):
        """
        Creates dataframe without columns in self.columns.
        :param x: Data.
        :param y: Labels.
        :return: Transformed dataframe.
        """
        return x[[c for c in x.columns if c not in self.columns]]


def add_category(data: pd.DataFrame):
    """
    Adds binary 'category' column to dataframe.
    :param data: Must include 'group' column.
    :return: Data with added column.
    """
    data["category"] = 0

    # train segments with more then 9 open channels classes
    data.loc[data.group == 3, 'category'] = 1

    return data


def read_input(data_path: str):
    """
    Reads .csv with data.
    :param data_path: Path to input data.
    :return: data - pd.DataFrame containing the input.
    """
    data = pd.read_csv(data_path, dtype={'time': np.float32, 'signal': np.float32, 'open_channels': np.uint8})
    return data


def save_submission(ss_path: str, out_path: str, y_test: np.ndarray):
    """
    Writes submission in format specified by competition host.
    :param ss_path: Path to sample submission.
    :param out_path: Path to output file.
    :param y_test: Numpy array with prediction.
    :return:
    """
    submission = pd.read_csv(ss_path)
    submission["open_channels"] = np.asarray(y_test, dtype=np.int32)
    submission.to_csv(out_path, index=False, float_format="%.4f")
