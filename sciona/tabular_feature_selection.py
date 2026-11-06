"""Explicit fold-local RFE policy for the pending binary tabular recipe."""
from dataclasses import dataclass
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from sciona.atoms.ml.sklearn.feature_selection.rfe_fit_bookkeeping.atoms import rfe_active_feature_indices


@dataclass(frozen=True)
class SelectedTabularFeatures:
    """Apply a fitted feature transformer followed by its fixed column subset."""
    source: object
    columns: tuple
    ranking: tuple

    def transform(self, numeric, categorical):
        return self.source.transform(numeric, categorical)[:, self.columns]


def select_training_features(features, matrix, labels, *, policy, seed):
    """Select on the supplied training matrix, never transform training again.

None explicitly disables selection. The RFE branch uses standardized logistic
coefficients, recursively removing the requested number of least-important
columns until the requested count remains. Scaling is refitted at each step.
Tie behavior follows the installed sklearn/NumPy implementation; cross-version
identity and causal relevance are not claimed. This is supervised recipe policy,
not a domain-specific importance interpretation or a validation-score search.
"""
    if policy is None:
        return features, matrix
    if type(policy) is not dict or set(policy) != {'method', 'keep', 'step'} or policy['method'] != 'rfe':
        raise ValueError('Explicit RFE method, keep count and step required')
    if (type(policy['keep']) is not int or not 1 <= policy['keep'] <= matrix.shape[1]
            or type(policy['step']) is not int or policy['step'] < 1):
        raise ValueError('RFE keep must fit available width; step must be positive')
    if matrix.shape[1] == 1:
        # No elimination is possible; sklearn RFE otherwise requires >=2 inputs.
        return SelectedTabularFeatures(features, (0,), (1,)), matrix.copy()
    estimator = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    selector = RFE(estimator, n_features_to_select=policy['keep'], step=policy['step'],
        importance_getter='named_steps.logisticregression.coef_')
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        selector.fit(matrix, labels)
    columns = tuple(int(i) for i in rfe_active_feature_indices(selector.support_))
    selected = SelectedTabularFeatures(features, columns, tuple(int(i) for i in selector.ranking_))
    return selected, np.asarray(matrix[:, columns], dtype=np.float64)
