import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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
    n_cuts: The number of cuts to create.
    cutpoints: The cutpoints to ceate.
    """
    def __init__(self, min=None, max=None, n_cuts=None,
                       n_params=None, cutpoints=None):
        if not cutpoints:
            if not n_cuts:
                n_cuts = n_params
            cutpoints = np.linspace(min, max, num=(n_cuts + 2))[1:-1] 
            max, min = np.max(cutpoints), np.min(cutpoints)
        self._max = max
        self._min = min
        self.cutpoints = np.asarray(cutpoints)

    @property
    def n_params(self):
        """
        Note: For fair accounting, we do NOT include the intercept term
        in the number of estimated parameters.
        """
        return len(self.cutpoints)

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        X_binned = self._transform_array(X)
        if isinstance(X, pd.Series):
            left_endpoints = ['neg_infinity'] + list(self.cutpoints)
            right_endpoints = list(self.cutpoints) + ['pos_infinity']
            col_names = [
                "{}_bin_{}_to_{}".format(X.name, le, re)
                for i, (le, re) in enumerate(zip(left_endpoints, right_endpoints))]
            X_binned = pd.DataFrame(X_binned, columns=col_names, index=X.index)
        return X_binned

    def _transform_array(self, X, **transform_params):
        X = X.squeeze()
        X_binned = np.empty((X.shape[0], self.n_params + 1))
        X_binned[:, 0] = X <= self.cutpoints[0]
        n_cutpoints = len(self.cutpoints)
        iter_cuts = enumerate(
            zip(self.cutpoints[:(n_cutpoints - 1)], self.cutpoints[1:]))
        for i, (left_cut, right_cut) in iter_cuts:
            X_binned[:, i+1] = (left_cut < X) & (X <= right_cut)
        X_binned[:, self.n_params] = self.cutpoints[-1] < X
        return X_binned


class Polynomial(BaseEstimator, TransformerMixin):
    """Apply a polynomial basis expansion to an array.

    Note that the array should be standardized before using this basis
    expansion.

    Parameters
    ----------
    degree: The degree of polynomial basis to use.
    """
    def __init__(self, degree=None, n_params=None):
        if not degree:
            degree = n_params
        self.degree = degree

    @property
    def n_params(self):
        return self.degree

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        X_poly = np.zeros((X.shape[0], self.degree))
        X_poly[:, 0] = X.squeeze()
        for i in range(1, self.degree):
            X_poly[:, i] = X_poly[:, i-1] * X.squeeze()
        return X_poly


class AbstractSpline(BaseEstimator, TransformerMixin):
    """Base class for all spline basis expansions."""
    def __init__(self, max=None, min=None, n_knots=None, n_params=None, knots=None):
        if not knots:
            if not n_knots:
               n_knots = self._compute_n_knots(n_params)
            knots = np.linspace(min, max, num=(n_knots + 2))[1:-1] 
            max, min = np.max(knots), np.min(knots)
        self.knots = np.asarray(knots)

    @property
    def n_knots(self):
        return len(self.knots)

    def fit(self, *args, **kwargs):
        return self


class LinearSpline(AbstractSpline):
    """Apply a piecewise linear basis expansion to an array.

    Create new features out of an array that can be used to fit a continuous
    piecewise linear function of the array.

    This transformer can be created by sepcifying the maximum, minimum, and
    number of knots, or by specifying the cutpoints directly.  If the knots are
    not directly sepcified, the resulting knots are equally space within the
    *interior* of (max, min).

    Parameters
    ----------
    min: Minimum of interval containing the knots.
    max: Maximum of the interval containing the knots.
    n_knots: The number of knots to create.
    knots: The knots.
    """
    def _compute_n_knots(self, n_params):
        return n_params - 1 

    @property
    def n_params(self):
        return self.n_knots + 1

    def transform(self, X, **transform_params):
        X_pl = np.zeros((X.shape[0], self.n_knots + 1))
        X_pl[:, 0] = X.squeeze()
        for i, knot in enumerate(self.knots, start=1):
            X_pl[:, i] = np.maximum(0, X - knot).squeeze()
        return X_pl


class CubicSpline(AbstractSpline):
    """Apply a piecewise cubic basis expansion to an array.

    Create new features out of an array that can be used to fit a continuous
    piecewise cubic function of the array.  The fitted curve is continuous to
    the second order at all of the knots.

    This transformer can be created by sepcifying the maximum, minimum, and
    number of knots, or by specifying the cutpoints directly.  If the knots are
    not directly sepcified, the resulting knots are equally space within the
    *interior* of (max, min).

    Parameters
    ----------
    min: Minimum of interval containing the knots.
    max: Maximum of the interval containing the knots.
    n_knots: The number of knots to create.
    knots: The knots.
    """
    def _compute_n_knots(self, n_params):
        return n_params - 3

    @property
    def n_params(self):
        return self.n_knots + 3

    def transform(self, X, **transform_params):
        X_spl = np.zeros((X.shape[0], self.n_knots + 3))
        X_spl[:, 0] = X.squeeze()
        X_spl[:, 1] = X_spl[:, 0] * X_spl[:, 0]
        X_spl[:, 2] = X_spl[:, 1] * X_spl[:, 0]
        for i, knot in enumerate(self.knots, start=3):
            X_spl[:, i] = np.maximum(0, (X - knot)*(X - knot)*(X - knot)).squeeze()
        return X_spl


class NaturalCubicSpline(AbstractSpline):
    """Apply a natural cubic basis expansion to an array.

    Create new features out of an array that can be used to fit a continuous
    piecewise cubic function of the array.

    This transformer can be created by sepcifying the maximum, minimum, and
    number of knots, or by specifying the cutpoints directly.  If the knots are
    not directly sepcified, the resulting knots are equally space within the
    *interior* of (max, min).

    Parameters
    ----------
    min: Minimum of interval containing the knots.
    max: Maximum of the interval containing the knots.
    n_knots: The number of knots to create.
    knots: The knots.
    """
    def _compute_n_knots(self, n_params):
        return n_params

    @property
    def n_params(self):
        return self.n_knots - 1

    def transform(self, X, **transform_params):
        X_spl = self._transform_array(X)
        if isinstance(X, pd.Series):
            col_names = ["{}_spline_linear".format(X.name)] + [
                "{}_spline_{}".format(X.name, idx)
                for idx in range(self.n_knots - 2)]
            X_spl = pd.DataFrame(X_spl, columns=col_names, index=X.index)
        return X_spl

    def _transform_array(self, X, **transform_params):
        X = X.squeeze()
        X_spl = np.zeros((X.shape[0], self.n_knots - 1))
        X_spl[:, 0] = X.squeeze()

        def d(knot_idx, x):
            ppart = lambda t: np.maximum(0, t)
            cube = lambda t: t*t*t
            numerator = (cube(ppart(x - self.knots[knot_idx]))
                            - cube(ppart(x - self.knots[self.n_knots - 1])))
            denominator = self.knots[self.n_knots - 1] - self.knots[knot_idx]
            return numerator / denominator

        for i in range(0, self.n_knots - 2):
            X_spl[:, i+1] = (d(i, X) - d(self.n_knots - 2, X)).squeeze()
        return X_spl
