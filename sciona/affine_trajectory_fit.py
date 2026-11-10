"""Compatibility boundary for the reusable affine least-squares provider."""
from sciona.atoms.numerical.statistics.affine_fit import affine_least_squares


def fit(coordinates, values, *, rcond):
    """Fit aligned coordinates/values with an explicit rank cutoff and diagnostics."""
    return affine_least_squares(coordinates, values, rcond)
