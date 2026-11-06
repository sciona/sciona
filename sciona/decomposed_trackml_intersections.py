"""TrackML layer policy with invocation-local reusable intersection operations.

Policy adapted from edwinst/trackml_solution, BSD-2-Clause; see
 docs/licenses/TrackML-BSD-2-Clause.txt. Layer selection, correction hooks,
recursive retries, source miss sentinels and multiplicity remain here.
"""
import numpy as np
from sciona.decomposed_trackml_lifecycle import _bound
from sciona.trackml_intersections import Intersector


def intersector_class(cylinder_function, cap_function):
    if not callable(cylinder_function) or not callable(cap_function):
        raise ValueError('Two callable intersection providers required')

    class CylinderIntersector:
        """Calculates intersections between helices and cylinder layers.
        """

        def __init__(self, cylspec):
            self.cylspec = cylspec

        def intersectFromInside(self, cyl_id, v, dir):
            """Intersect straight rays from inside of the cylinder with the cylinder.
            Note: This function works for both 2-dimensional and 3-dimensional coordinates.
            XXX: Split arguments into coordinates?
            """
            r2 = self.cylspec.cylinders_df.iloc[cyl_id]['cyl_r2']
            v2 = v[:, 0:2]
            r2sqrd = np.einsum('ij,ij->i', v2, v2) - r2 ** 2
            assert np.all(r2sqrd < 0)
            dir2 = dir[:, 0:2]
            dirnorm = dir / np.linalg.norm(dir2, axis=1)[:, np.newaxis]
            dir2norm = dirnorm[:, 0:2]
            v2in = np.einsum('ij,ij->i', v2, dir2norm)
            disc = np.square(v2in) - r2sqrd
            assert np.all(disc > 0)
            alpha = -v2in + np.sqrt(disc)
            ints = v + alpha[:, np.newaxis] * dirnorm
            return ints

        def intersectHelices(self, x0, y0, z0, uz0, hel_xm, hel_ym, hel_r, hel_pitch, iterations=10, pre_move=10.0, missable=True, corrector=None):
            """Intersect helical trajectories with the next cylinder hit by the respective trajectory.
            Args:
                x0, y0, z0 (array (n_points,)): coordinates of initial vertices of the trajectories
                uz0 (array (n_points,)): estimated tangent vector z-component of the trajectories
                    Note: Only the sign of uz0 is used in the calculation.
                hel_xm, hel_ym (array (n_points,)): center coordinates of helices in the x,y-plane
                hel_r (array (n_points,)): radii of the helices in the x,y-plane
                hel_pitch (array (n_points,)): pitches of the helices along the z-axis
                iterations (positive int): maximum number of iterations of intersection finding to do
                    Note: Multiple iterations are needed when helices intersect cylinders in the
                          x,y-plane but miss the cylinder in z-coordinate.
                pre_move (float or array (n_points,)): minimum distance to move (when projected
                    to the x,y-plane) along the helix before considering the next intersection
                    May be negative if the intention is to hit the intersection point again.
                missable...if True, detect cases where a layer is laterally missed, and in these cases
                    try to find the following layer intersection
                corrector (None or HelixCorrector): if given, apply this helix corrector to correct
                    the predicted intersection points
            Returns:
                xi (array (n_points,)): x-coordinate of estimated next hit, or np.nan if none
                yi (array (n_points,)): y-coordinate of estimated next hit, or np.nan if none
                zi (array (n_points,)): z-coordinate of estimated next hit, or np.nan if none
                cyl_id (integer array (n_points,)): cyl_id of estimated next hit, or -1 if none
                hel_dphi (array (n_points,)): helix phase difference to next hit, or np.nan if none
                hel_s (array (n_points,)): positive arc length until next hit, or np.inf if none
                mult (integer array (n_points,)): estimated maximum multiplicity of the intersection
            """
            dx = x0 - hel_xm
            dy = y0 - hel_ym
            phi0 = np.arctan2(dy, dx)
            hel_rm = np.sqrt(np.square(hel_xm) + np.square(hel_ym))
            phim0 = phi0 - np.arctan2(hel_ym, hel_xm)
            phim0[phim0 < 0] += 2 * np.pi
            phim0[phim0 >= 2 * np.pi] -= 2 * np.pi
            max_r2sqr = np.square(hel_rm + hel_r)
            min_r2sqr = np.square(hel_rm - hel_r)
            max_bin = np.digitize(max_r2sqr, self.cylspec.cyl_rsqr)
            min_bin = np.digitize(min_r2sqr, self.cylspec.cyl_rsqr)
            dphi_tolerance = pre_move / hel_r
            hel_rm_sqr_plus_hel_r_sqr = np.square(hel_rm) + np.square(hel_r)
            two_hel_rm_hel_r = 2 * hel_rm * hel_r
            r2sqr0 = hel_rm_sqr_plus_hel_r_sqr + two_hel_rm_hel_r * np.cos(phim0 + np.sign(uz0 * hel_pitch) * dphi_tolerance)
            current_bin = np.digitize(r2sqr0, self.cylspec.cyl_rsqr)
            can_intersect = max_bin > min_bin
            moving_outward = z0 * uz0 > 0
            if missable:
                can_intersect[moving_outward & (np.abs(z0) > np.take(self.cylspec.cyl_absz_cummax, max_bin - 1, mode='clip'))] = False
            sign_r2sqr_dot = np.sign(-hel_pitch * uz0 * np.sin(phim0))
            where_zero = np.where(sign_r2sqr_dot == 0)[0]
            sign_r2sqr_dot[where_zero] = np.sign(-np.cos(phim0[where_zero]))
            sign_r2sqr_dot[current_bin == max_bin] = -1
            sign_r2sqr_dot[current_bin == min_bin] = 1
            next_cyl_id = (current_bin + (sign_r2sqr_dot - 1) / 2).astype(np.int8)
            if missable:
                min_cyl_id_bins = np.digitize(np.abs(z0[moving_outward]), self.cylspec.cyl_absz_cuts).astype(np.int8)
                next_cyl_id[moving_outward] = np.maximum(next_cyl_id[moving_outward], self.cylspec.cyl_absz_min_cyl_id[min_cyl_id_bins])
            next_cyl_id[~can_intersect] = 0
            next_r2sqr = self.cylspec.cyl_rsqr[next_cyl_id]
            can_intersect &= next_r2sqr <= max_r2sqr
            can_intersect &= next_r2sqr >= min_r2sqr
            xi, yi, zi, dphi, numeric_hit = cylinder_function(
                x0, y0, z0, hel_xm, hel_ym, hel_r, hel_pitch, uz0,
                next_r2sqr, np.broadcast_to(dphi_tolerance, x0.shape))
            can_intersect &= numeric_hit
            hel_p = hel_pitch / (2 * np.pi)
            if corrector is not None:
                xi, yi, zi = corrector.correctCylinderIntersections(xi, yi, zi, next_cyl_id, mask=can_intersect)
            hel_s = np.sqrt(np.square(hel_r) + np.square(hel_p)) * np.abs(dphi)
            mult = np.full(x0.shape[0], 4, dtype=np.int8)
            if missable:
                is_beyond = moving_outward & can_intersect & (np.abs(zi) > self.cylspec.cyl_absz_max[next_cyl_id])
                if np.any(is_beyond) and iterations > 1:
                    x0_next = xi[is_beyond]
                    y0_next = yi[is_beyond]
                    z0_next = zi[is_beyond]
                    if corrector is not None:
                        corrector_next = corrector.subset(is_beyond)
                        hel_params_next = corrector_next.helixParams()
                    else:
                        corrector_next = None
                        hel_params_next = (par[is_beyond] for par in (uz0, hel_xm, hel_ym, hel_r, hel_pitch))
                    xi_next, yi_next, zi_next, cyl_id_next, dphi_next, hel_s_next, mult_next = self.intersectHelices(x0_next, y0_next, z0_next, *hel_params_next, corrector=corrector_next, iterations=iterations - 1)
                    np.place(xi, is_beyond, xi_next)
                    np.place(yi, is_beyond, yi_next)
                    np.place(zi, is_beyond, zi_next)
                    np.place(next_cyl_id, is_beyond, cyl_id_next)
                    dphi[is_beyond] += dphi_next
                    hel_s[is_beyond] += hel_s_next
                    np.place(mult, is_beyond, mult_next)
                else:
                    can_intersect &= ~is_beyond
            xi[~can_intersect] = np.nan
            yi[~can_intersect] = np.nan
            zi[~can_intersect] = np.nan
            next_cyl_id[~can_intersect] = -1
            dphi[~can_intersect] = np.nan
            hel_s[~can_intersect] = np.inf
            mult[~can_intersect] = 0
            return (xi, yi, zi, next_cyl_id, dphi, hel_s, mult)

    class CapIntersector:
        """Calculates intersections between helices and cap layers.
        """

        def __init__(self, capspec):
            self.capspec = capspec

        def intersectHelices(self, x0, y0, z0, uz0, hel_xm, hel_ym, hel_r, hel_pitch, iterations=10, pre_move=50.0, missable=True, corrector=None):
            """Intersect helical trajectories with the next cap hit by the respective trajectory.
            Args:
                x0, y0, z0 (array (n_points,)): coordinates of initial vertices of the trajectories
                uz0 (array (n_points,)): estimated tangent vector z-component of the trajectories
                    Note: Only the sign of uz0 is used in the calculation.
                hel_xm, hel_ym (array (n_points,)): center coordinates of helices in the x,y-plane
                hel_r (array (n_points,)): radii of the helices in the x,y-plane
                hel_pitch (array (n_points,)): pitches of the helices along the z-axis
                iterations (positive int): maximum number of iterations of intersection finding to do
                    Note: Multiple iterations are needed when helices pass z-coordinates of caps
                          but miss the cap radially.
                pre_move (float or array (n_points,)): minimum distance in z-coordinate to move
                    along the helix before considering an intersection. May be negative if the
                    intention is to hit the previous intersection point again.
                missable...if True, detect cases where a layer is laterally missed, and in these cases
                    try to find the following layer intersection
            Returns:
                xi (array (n_points,)): x-coordinate of estimated next hit, or np.nan if none
                yi (array (n_points,)): y-coordinate of estimated next hit, or np.nan if none
                zi (array (n_points,)): z-coordinate of estimated next hit, or np.nan if none
                cap_id (integer array (n_points,)): cap_id of estimated next hit, or -1 if none
                hel_dphi (array (n_points,)): helix phase difference to next hit, or np.nan if none
                hel_s (array (n_points,)): positive arc length until next hit, or np.inf if none
                mult (integer array (n_points,)): estimated maximum multiplicity of the intersection
                corrector (None or HelixCorrector): if given, apply this helix corrector to correct
                    the predicted intersection points
            """
            sign_uz0 = np.sign(uz0)
            binning_z = z0 + sign_uz0 * pre_move
            bins = np.digitize(binning_z, self.capspec.cap_z)
            next_cap_id = (bins + (sign_uz0 - 1) / 2).astype(np.int8)
            can_hit = (sign_uz0 != 0) & (next_cap_id >= 0) & (next_cap_id < len(self.capspec.cap_z))
            where_can_hit = np.where(can_hit)[0]
            next_cap_id_ch = next_cap_id.compress(can_hit)
            next_z = self.capspec.cap_z[next_cap_id_ch]
            hel_p_ch = hel_pitch.compress(can_hit) / (2 * np.pi)
            if next_z.size:
                next_x, next_y, returned_z, dphi = cap_function(
                    x0.compress(can_hit), y0.compress(can_hit), z0.compress(can_hit),
                    hel_xm.compress(can_hit), hel_ym.compress(can_hit),
                    hel_r.compress(can_hit), hel_pitch.compress(can_hit), next_z)
                if not np.array_equal(returned_z, next_z):
                    raise ValueError('Plane provider changed target height')
            else:
                next_x, next_y, dphi = (np.empty(0, dtype=np.float64) for _ in range(3))
            if corrector is not None:
                corrector_ch = corrector.subset(can_hit)
                next_x, next_y = corrector_ch.correctCapIntersections(next_x, next_y, next_z, next_cap_id_ch)
            xi = np.full(x0.shape, np.nan)
            yi = np.full(y0.shape, np.nan)
            zi = np.full(y0.shape, np.nan)
            cap_id = np.full(x0.shape, -1, dtype=np.int8)
            xi[where_can_hit] = next_x
            yi[where_can_hit] = next_y
            zi[where_can_hit] = next_z
            cap_id[where_can_hit] = next_cap_id_ch
            hel_dphi = np.full(x0.shape, np.nan)
            hel_dphi[where_can_hit] = dphi
            hel_s_ch = np.sqrt(np.square(hel_r.compress(can_hit)) + np.square(hel_p_ch)) * np.abs(dphi)
            hel_s = np.full(x0.shape, np.inf)
            hel_s[where_can_hit] = hel_s_ch
            ri2_sqr_ch = np.square(next_x) + np.square(next_y)
            mult = np.zeros(x0.shape, dtype=np.int8)
            mult[where_can_hit] = 3
            for r2_min, r2_max in zip(self.capspec.ring_overlap_r2_min_sqr, self.capspec.ring_overlap_r2_max_sqr):
                mult[where_can_hit] += 1 * ((ri2_sqr_ch > r2_min) & (ri2_sqr_ch < r2_max))
            if missable:
                cap_r2_max_sqr_ch = self.capspec.caps_r2_max_sqr[next_cap_id_ch]
                cap_r2_min_sqr_ch = self.capspec.caps_r2_min_sqr[next_cap_id_ch]
                is_beyond = (ri2_sqr_ch < cap_r2_min_sqr_ch) | (ri2_sqr_ch > cap_r2_max_sqr_ch)
                for r2_min, r2_max in zip(self.capspec.ring_gap_r2_min_sqr, self.capspec.ring_gap_r2_max_sqr):
                    is_beyond |= (ri2_sqr_ch > r2_min) & (ri2_sqr_ch < r2_max)
                where_is_beyond = where_can_hit[is_beyond]
                if where_is_beyond.size > 0 and iterations > 1:
                    x0_next = next_x[is_beyond]
                    y0_next = next_y[is_beyond]
                    z0_next = next_z[is_beyond]
                    if corrector is not None:
                        corrector_next = corrector_ch.subset(is_beyond)
                        hel_params_next = corrector_next.helixParams()
                    else:
                        corrector_next = None
                        hel_params_next = (par[where_is_beyond] for par in (uz0, hel_xm, hel_ym, hel_r, hel_pitch))
                    xi_next, yi_next, zi_next, cap_id_next, dphi_next, hel_s_next, mult_next = self.intersectHelices(x0_next, y0_next, z0_next, *hel_params_next, corrector=corrector_next, iterations=iterations - 1)
                    xi[where_is_beyond] = xi_next
                    yi[where_is_beyond] = yi_next
                    zi[where_is_beyond] = zi_next
                    cap_id[where_is_beyond] = cap_id_next
                    hel_dphi[where_is_beyond] += dphi_next
                    hel_s[where_is_beyond] += hel_s_next
                    mult[where_is_beyond] = mult_next
                else:
                    xi[where_is_beyond] = np.nan
                    yi[where_is_beyond] = np.nan
                    zi[where_is_beyond] = np.nan
                    cap_id[where_is_beyond] = -1
                    hel_dphi[where_is_beyond] = np.nan
                    hel_s[where_is_beyond] = np.inf
                    mult[where_is_beyond] = 0
            return (xi, yi, zi, cap_id, hel_dphi, hel_s, mult)


    class BoundIntersector(Intersector):
        pass

    BoundIntersector.__init__ = _bound(Intersector.__init__,
        dict(CylinderIntersector=CylinderIntersector, CapIntersector=CapIntersector))
    return BoundIntersector
