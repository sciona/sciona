"""Inject reusable decision arithmetic while retaining the source scoring recipe.

BSD-2-Clause adaptation; see docs/licenses/TrackML-BSD-2-Clause.txt.
Coordinate projection, empirical error prediction and layer lookup remain recipe
operations. The supplied decision function receives explicit dimensionless
thresholds and aligned numeric vectors, without event or detector state.
"""
import numpy as np

from sciona.decomposed_trackml_lifecycle import _bound
from sciona.decomposed_trackml_neighbor_lifecycle import execute as neighbor_execute
from sciona.decomposed_trackml_correction_lifecycle import lifecycle_class


def scorer_class(base, decision_function):
    if not callable(decision_function):
        raise ValueError('Callable decision provider required')

    class Scorer(base):
        def bayesianNeighborEvaluation(self, run, nb_df, mask=None, step=0, details=None):
            x, y, z = run.event.hitCoordinatesById(nb_df['extend_hit_id'])
            d_theta, d_phi = self.projectNeighborDisplacement(
                x, y, z, nb_df['xn'].values, nb_df['yn'].values, nb_df['zn'].values)
            b_theta, b_phi, measured_theta, measured_phi = run.neighbors.evaluateLayerFunctions(
                x, y, z, nb_df['cyl_closer'].values, nb_df['next_id'].values,
                functions=('d_utheta_abs', 'd_uphi_abs', 'dnt_utheta_abs', 'dnt_uphi_abs'))
            e_theta, e_phi = self.predictRandomError(
                run, hel_s=nb_df['hel_s'], e_theta_meas=measured_theta, e_phi_meas=measured_phi)
            values = tuple(np.asarray(a) for a in (d_theta, d_phi, b_theta, b_phi, e_theta, e_phi))
            if any(a.dtype != np.float64 or a.shape != (len(nb_df),) or
                   not np.isfinite(a).all() for a in values):
                raise ValueError('Decision requires aligned finite float64 vectors')
            if any(np.any(a <= 0) for a in values[4:]):
                raise ValueError('Decision error scales must be positive')
            cut_factor, trust = run.params['nb__cut_factor'], run.params['nb__dist_trust']
            for threshold in (cut_factor, trust):
                if isinstance(threshold, (bool, np.bool_)) or not np.isscalar(threshold) or not np.isfinite(threshold) or threshold <= 0:
                    raise ValueError('Decision thresholds must be finite positive scalars')
            # Preserve the source evaluation order and diagnostic values. These
            # diagnostics are deliberately not reconstructed from boolean flags.
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                de = np.sqrt(np.square(values[0] / values[4]) + np.square(values[1] / values[5]))
                cut = cut_factor * np.sqrt(2 * np.log(
                    (2 * np.square(values[2] / values[4]) / np.pi + 1) *
                    (2 * np.square(values[3] / values[5]) / np.pi + 1)))
                trusted_cut = trust * cut
            if not all(np.isfinite(a).all() for a in (de, cut, trusted_cut)):
                raise ValueError('Decision arithmetic must remain finite')
            if len(nb_df):
                result = decision_function(*values, cut_factor=cut_factor, dist_trust=trust)
            else:
                # The source accepts an empty frame; the provider requires N>0.
                result = (np.empty(0, dtype=bool), np.empty(0, dtype=bool))
            if details:
                details.bay_index = nb_df['extend_index'].values
                details.bay_hit_id = nb_df['extend_hit_id'].values
                for name, value in zip(('b_theta', 'b_phi', 'd_theta', 'd_phi', 'e_theta', 'e_phi'),
                                       (b_theta, b_phi, d_theta, d_phi, e_theta, e_phi)):
                    setattr(details, 'bay_' + name, value)
                details.bay_cut, details.bay_de = cut, de
            return result

    return Scorer


def execute(*args, decision_function, **kwargs):
    def local_class(*providers):
        return scorer_class(lifecycle_class(*providers), decision_function)
    return _bound(neighbor_execute, {'correction_class': local_class})(*args, **kwargs)
