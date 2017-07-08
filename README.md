# Basis Expansions

This module includes some scikit-learn style transformers that perform basis expansions on a feature `x` for use in regression.

The following classes are included:

  - `Binner`: Creates indicator variables by segmenting the range of `x` into disjoint intervals.
  - `Polynomial`: Creates polynomial features.  Using these features in a regression fits a polynomial function of `x`.
  - `LinearSpline`: Creates a piecewise linear spline which joins continuously at the knots.
  - `CubicSpline`: Creates a piecewise cubic spline which joins continuously, differentiably, and second differentiably at the knots.
  - `NaturalCubicSpline`: Creates a piecewise natural cubic spline (cubic curves in the interior segments, linear in the exterior segments) which joins continuously, differentiably, and second differentiably at the knots.

 ![Basis Transformations](img/basis-transformations.png)

 See the `basis-expansions-regressions.ipynb` notebook for examples of use.
