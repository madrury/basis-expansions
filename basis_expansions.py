import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnSelector(object):
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


class Binner(BaseEstimator, TransformerMixin):
    """Apply a binning basis expansion to an array.

    Create new features out of an array by binning the values of that array
    based on a sequence of intervals. An indicator feature is created for each
    bin, indicating which bin the given observation falls into.
    
    This transformer can be created by sepcifying the maximum, minimum, and 
    number of cutpoints, or by specifying the cutpoints directly.

    Parameters
    ----------
    min: Minimum cutpoint for the bins.
    max: Maximum cutpoint for the bins.
    n_bins: The number of bins to create.
    cutpoints: The cutpoints to ceate.
    """

    def __init__(self, min=None, max=None, n_bins=None, cutpoints=None):
        self._min = min
        self._max = max
        if not cutpoints:
            if n_bins < 2:
                raise ValueError("n_bins must be >= 2")
            cutpoints = np.linspace(min, max, num=(n_bins - 1)) 
        self._n_bins = len(cutpoints) + 1
        self._cutpoints = cutpoints

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        X = X.squeeze()
        X_binned = np.empty((X.shape[0], self._n_bins))
        X_binned[:, 0] = X <= self._cutpoints[0]
        n_cutpoints = len(self._cutpoints)
        iter_cuts = enumerate(
            zip(self._cutpoints[:(n_cutpoints - 1)], self._cutpoints[1:]))
        for i, (left_cut, right_cut) in iter_cuts:
            X_binned[:, i+1] = (left_cut < X) * (X <= right_cut)
        X_binned[:, self._n_bins - 1] = self._cutpoints[-1] < X
        return X_binned


class Polynomial(object):
    """Apply a polynomial basis expansion to an array.

    Note that the array should be standardized before using this basis
    expansion.

    Parameters
    ----------
    degree: The degree of polynomial basis to use.
    """

    def __init__(self, degree):
        self.degree = degree

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        X_poly = np.zeros((X.shape[0], self.degree))
        X_poly[:, 0] = X.squeeze()
        for i in range(1, self.degree):
            X_poly[:, i] = X_poly[:, i-1] * X.squeeze()
        return X_poly
