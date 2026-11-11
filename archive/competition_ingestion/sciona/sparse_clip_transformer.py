"""Pipeline adapter for registered clipping with explicit feature coordinates.

Slot identifiers refer to one fitted upstream feature pipeline. They do not
establish equivalence of independently fitted vocabularies. The caller supplies
a coordinate identifier and retains the entire upstream fitted state.
"""
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix
from sciona.atoms.ml.fitted_sparse_clipping import fit_sparse_clip, apply_sparse_clip


class SparseClipTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, feature_coordinate):
        self.feature_coordinate = feature_coordinate
        self.state_ = None

    @staticmethod
    def canonical(values):
        if not isinstance(values, csr_matrix):
            raise ValueError('CSR feature matrix required')
        result = values.copy()
        result.sort_indices()
        if not result.has_canonical_format:
            raise ValueError('Duplicate feature entries require an explicit upstream adapter')
        return result

    def fit(self, values, y=None):
        self.state_ = None
        columns = [f'pipeline-slot-{i}' for i in range(values.shape[1])]
        self.state_ = fit_sparse_clip(self.canonical(values), columns, self.feature_coordinate)
        return self

    def transform(self, values):
        if self.state_ is None:
            raise ValueError('Complete fitted clipping state required')
        return apply_sparse_clip(self.canonical(values), self.state_['feature_columns'],
                                 self.feature_coordinate, self.state_)
