import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.metaestimators import _BaseComposition


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Transformer that selects a column in a numpy array or DataFrame
    by index or name.
    """
    def __init__(self, idxs=None, name=None):
        self.idxs = np.asarray(idxs)

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        # Need to teat pandas data frames and numpy arrays slightly differently.
        if isinstance(X, pd.DataFrame) and idxs:
            return X.iloc[:, self.idxs]
        if isinstance(X, pd.DataFrame) and name:
            return X[name]
        return X[:, self.idxs]


class FeatureUnion(_BaseComposition, TransformerMixin):

    def __init__(self, transformer_list):
        self.transformer_list = transformer_list

    def fit(self, X, y=None):
        for t in self.transformer_list:
            t.fit(X, y)

    def transform(self, X, *args, **kwargs):
        Xs = [t.transform(X) for t in self.transformer_list]
        if isinstance(X, pd.DataFrame):
            return pd.concat(Xs, axis=1)
        return np.hstack(Xs)
