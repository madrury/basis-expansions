"""
basis_expansions: Basis Expansions for Regression.

The basis_expansions module contains classes for basis expansions to be used in
regression models.  Given a feature x, a basis expansions for that feature x is
a collection of functions

    f_0, f_1, f_2, ...

that are meant to be applied to the feature to construct derived features in a
regression model.  The functions in the expansions are often chosen to allow
the model to adapt to non-linear shapes in the predictor/response relationship.

Each class in this module conforms to the scikit-learn transformer api, and
work on both numpy.array and pandas.Series objects.

The following basis expansions are supported:
    - Binner: Cut the range of x into bins, and create indicator features for
      bin membership.
    - Polynomial: Polynomial expansion of a given degree.
    - LinearSpline: Piecewise linear spline.
    - CubicSpline: Piecewise cubic spline.
    - NaturalCubicSpline: Piecewise cubic spline constrained to be linear
      outside of knots.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class Binner(BaseEstimator, TransformerMixin):
    """Apply a binning basis expansion to an array.

    Creates new features out of an array by binning the values of that array
    based on a sequence of intervals. An indicator feature is created for each
    bin, indicating which bin the given observation falls into.
    
    This transformer can be created in two ways:
      - By specifying the maximum, minimum, and number of cutpoints (or the
        number of parameters to estimate).
      - By specifying the cutpoints directly.

    Parameters
    ----------
    min: float
        Minimum cutpoint for the bins.
    
    max: float
        Maximum cutpoint for the bins.
    
    n_cuts: 
        The number of cuts to create.

    n_params:
        The number of non-intercept parameters to estimate in a regression
        using the transformed features.  Equal to the number of cutpoints.
    
    cutpoints: 
        The cutpoints to use in the bins.
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
        """The number of parameters estimated when regression on features
        created by Binner.

        Notes
        -----
        For fair accounting, we do NOT include the intercept term in the number
        of estimated parameters.
        """
        return len(self.cutpoints)

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        X_binned = self._transform_array(X)
        if isinstance(X, pd.Series):
            col_names = self._make_names()
            X_binned = pd.DataFrame(X_binned, columns=col_names, index=X.index)
        return X_binned

    def _make_names(self):
        left_endpoints = ['neg_infinity'] + list(self.cutpoints)
        right_endpoints = list(self.cutpoints) + ['pos_infinity']
        col_names = [
            "{}_bin_{}_to_{}".format(X.name, le, re)
            for i, (le, re) in enumerate(zip(left_endpoints, right_endpoints))]
        return col_names

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


    Parameters
    ----------
    degree: positive integer.
        The degree of polynomial basis to use.

    n_params:
        The number of non-intercept parameters to estimate in a regression
        using the transformed features.  Equal to the degree.

    Notes
    -----
    The array should be standardized before using this basis expansion to void
    numerical issues.
    """
    def __init__(self, degree=None, n_params=None):
        if not degree:
            degree = n_params
        self.degree = degree

    @property
    def n_params(self):
        """The number of parameters estimated when regression on features
        created by Polynomial.
        """
        return self.degree

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        X_poly = self._transform_array(X)
        if isinstance(X, pd.Series):
            col_names = [
                "{}_degree_{}".format(X.name, d) for d in range(1, self.degree + 1)]
            X_poly = pd.DataFrame(X_poly, columns=col_names, index=X.index)
        return X_poly

    def _transform_array(self, X, **transform_params):
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

    The features created with this basis expansion can be used to fit a
    piecewise linear function.  Exact form of the basis functions are:

        f_0(x) = x
        f_1(x) = max(0, x - k_1)
        ...
        f_j(x) = max(0, x - k_j)

    This transformer can be created in two ways:
      - By specifying the maximum, minimum, and number of knots. 
      - By specifying the knots directly.  
      
    If the knots are not directly specified, the resulting knots are equally
    space within the *interior* of (max, min).  That is, the endpoints are
    *not* included as knots.

    Parameters
    ----------
    min: float 
        Minimum of interval containing the knots.

    max: float 
        Maximum of the interval containing the knots.

    n_knots: positive integer 
        The number of knots to create.

    knots: array or list of floats 
        The knots.
    """
    def _compute_n_knots(self, n_params):
        return n_params - 1 

    @property
    def n_params(self):
        return self.n_knots + 1

    def transform(self, X, **transform_params):
        X_pl = self._transform_array(X, **transform_params)
        if isinstance(X, pd.Series):
            col_names = self._make_names(X)
            X_pl = pd.DataFrame(X_pl, columns=col_names, index=X.index)
        return X_pl

    def _make_names(self, X):
        first_name = "{}_spline_linear".format(X.name)
        rest_names = ["{}_spline_{}".format(X.name, idx)
                      for idx in range(self.n_knots)]
        return [first_name] + rest_names

    def _transform_array(self, X, **transform_params):
        X_pl = np.zeros((X.shape[0], self.n_knots + 1))
        X_pl[:, 0] = X.squeeze()
        for i, knot in enumerate(self.knots, start=1):
            X_pl[:, i] = np.maximum(0, X - knot).squeeze()
        return X_pl


class CubicSpline(AbstractSpline):
    """Apply a piecewise cubic basis expansion to an array.

    
    The features created with this basis expansion can be used to fit a
    piecewise cubic function.  The fitted curve is continuously differentiable
    to the second order at all of the knots.

    This transformer can be created in two ways:
      - By specifying the maximum, minimum, and number of knots.
      - By specifying the cutpoints directly.  
      
    If the knots are not directly specified, the resulting knots are equally
    space within the *interior* of (max, min).  That is, the endpoints are
    *not* included as knots.

    Parameters
    ----------
    min: float 
        Minimum of interval containing the knots.

    max: float 
        Maximum of the interval containing the knots.

    n_knots: positive integer 
        The number of knots to create.

    knots: array or list of floats 
        The knots.
    """
    def _compute_n_knots(self, n_params):
        return n_params - 3

    @property
    def n_params(self):
        return self.n_knots + 3

    def transform(self, X, **transform_params):
        X_spl = self._transform_array(X)
        if isinstance(X, pd.Series):
            col_names = self._make_names(X)
            X_spl = pd.DataFrame(X_spl, columns=col_names, index=X.index)
        return X_spl

    def _make_names(self, X):
        first_name = "{}_spline_linear".format(X.name)
        second_name = "{}_spline_quadratic".format(X.name)
        third_name = "{}_spline_cubic".format(X.name)
        rest_names = ["{}_spline_{}".format(X.name, idx)
                      for idx in range(self.n_knots)]
        return [first_name, second_name, third_name] + rest_names

    def _transform_array(self, X, **transform_params):
        X_spl = np.zeros((X.shape[0], self.n_knots + 3))
        X_spl[:, 0] = X.squeeze()
        X_spl[:, 1] = X_spl[:, 0] * X_spl[:, 0]
        X_spl[:, 2] = X_spl[:, 1] * X_spl[:, 0]
        for i, knot in enumerate(self.knots, start=3):
            X_spl[:, i] = np.maximum(0, (X - knot)*(X - knot)*(X - knot)).squeeze()
        return X_spl


class NaturalCubicSpline(AbstractSpline):
    """Apply a natural cubic basis expansion to an array.

    The features created with this basis expansion can be used to fit a
    piecewise cubic function under the constraint that the fitted curve is
    linear *outside* the range of the knots..  The fitted curve is continuously
    differentiable to the second order at all of the knots.

    This transformer can be created in two ways:
      - By specifying the maximum, minimum, and number of knots.
      - By specifying the cutpoints directly.  
      
    If the knots are not directly specified, the resulting knots are equally
    space within the *interior* of (max, min).  That is, the endpoints are
    *not* included as knots.

    Parameters
    ----------
    min: float 
        Minimum of interval containing the knots.

    max: float 
        Maximum of the interval containing the knots.

    n_knots: positive integer 
        The number of knots to create.

    knots: array or list of floats 
        The knots.
    """
    def _compute_n_knots(self, n_params):
        return n_params

    @property
    def n_params(self):
        return self.n_knots - 1

    def transform(self, X, **transform_params):
        X_spl = self._transform_array(X)
        if isinstance(X, pd.Series):
            col_names = self._make_names(X)
            X_spl = pd.DataFrame(X_spl, columns=col_names, index=X.index)
        return X_spl

    def _make_names(self, X):
        first_name = "{}_spline_linear".format(X.name)
        rest_names = ["{}_spline_{}".format(X.name, idx)
                     for idx in range(self.n_knots - 2)]
        return [first_name] + rest_names

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
