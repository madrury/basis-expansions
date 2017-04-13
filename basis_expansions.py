import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Binner(BaseEstimator, TransformerMixin):

    def __init__(self, min, max, cutpoints=None, n_bins=None):
        self._min = min
        self._max = max
        if not cutpoints:
            if n_bins < 2:
                raise ValueError("n_bins must be >= 2")
            cutpoints = np.linspace(min, max, num=(n_bins - 2)) 
        self._n_bins = 
        self._cutpoints = cutpoints

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.squeeze()
        X_binned = np.empty((X.shape[0], self._n_bins))
        X_binned[:, 0] = X <= self._cutpoints[0]
        n_cutpoints = len(self._cutpoints)
        iter_cuts = enumerate(
            zip(self._cutpoints[:(n_cutpoints - 1)], self._cutpoints[1:]))
        for i, (left_cut, right_cut) in iter_cuts:
            X_binned[:, i+1] = left_cut < X <= right_cut
        X_binned[:, self._n_bins] = self._cutpoints[-1] < X
        return X_binned
