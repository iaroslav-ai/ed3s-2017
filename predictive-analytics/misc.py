from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import numpy as np


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Selects a single column with index `key` from some matrix X"""
    def __init__(self, key, row_space=False):
        self.key = key
        self.row_space = row_space

    def fit(self, X, y=None):
        return self  # do nothing during fitting procedure

    def transform(self, data_matrix):
        if self.row_space:
            # select over first dimension
            return np.array(data_matrix[[self.key]])
        else:
            # select over second dimension
            return np.array(data_matrix[:, [self.key]])


class IntEncoder(BaseEstimator, TransformerMixin):
    """Assumes that input X to fit and transform is a single
    column matrix of categorical values."""
    def fit(self, X, y=None):
        # create label encoder
        v = X[:, 0].tolist()
        unique = set(v)
        self.decoder = dict(enumerate(unique))
        self.encoder = {v: k for k, v in self.decoder.items()}
        return self

    def transform(self, X, y=None):
        return np.array([[self.encoder[v]] for v in X[:, 0]])

    def inverse_transform(self, X, y=None):
        return np.array([self.decoder[v] for v in X[:, 0]])