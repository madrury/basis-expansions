import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Transformer that selects a column in a numpy array or DataFrame
    by index or name.
    """
    def __init__(self, idxs=None, name=None):
        self.idxs = np.asarray(idxs)
        self.idxs = idxs
        self.name = name

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        # Need to teat pandas data frames and numpy arrays slightly differently.
        if isinstance(X, pd.DataFrame) and self.idxs:
            return X.iloc[:, self.idxs]
        if isinstance(X, pd.DataFrame) and self.name:
            return X[self.name]
        return X[:, self.idxs]


class FeatureUnion(TransformerMixin):

    def __init__(self, transformer_list):
        self.transformer_list = transformer_list

    def fit(self, X, y=None):
        for _, t in self.transformer_list:
            t.fit(X, y)

    def transform(self, X, *args, **kwargs):
        Xs = [t.transform(X) for _, t in self.transformer_list]
        if isinstance(X, pd.DataFrame):
            return pd.concat(Xs, axis=1)
        return np.hstack(Xs)


class Intercept(TransformerMixin):

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, *args, **kwargs):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            return pd.Series(np.ones(X.shape[0]),
                             index=X.index, name="intercept")
        return np.ones(X.shape[0])
