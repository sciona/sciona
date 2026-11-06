"""Numeric sparse randomized projection, independent of vocabulary or labels."""
import numpy as np
from scipy import linalg, sparse
from sklearn.utils.extmath import randomized_range_finder, svd_flip
from threadpoolctl import threadpool_limits


def _matrix(matrix):
    if not isinstance(matrix,sparse.csr_matrix) or matrix.dtype!=np.float64:
        raise ValueError('Float64 CSR matrix required')
    x=matrix.copy();x.check_format(full_check=True)
    if not np.isfinite(x.data).all():raise ValueError('Finite CSR required')
    return x


def fit_projection(matrix, rank, iterations, oversamples, seed):
    """Return right singular vectors and singular values from finite float64 CSR.

    No centering or scaling. Rank is positive and less than both dimensions;
    iteration and oversampling counts are positive integers. Uses the smaller
    orientation, sklearn's auto-normalized randomized range finder and SciPy
    gesdd. Right-vector sign convention matches sklearn TruncatedSVD. The
    dense basis lift uses explicit einsum to avoid the observed layout-sensitive
    matmul warning. Sparse input is copied, never densified as a full matrix;
    reduced basis/projection and returned factors are dense. CSR entry ordering
    and duplicate storage are preserved on the copy. Caller owns total
    concurrency, feature ordering, fitting populations and downstream policy.
    """
    x=_matrix(matrix)
    if any(type(v) is not int for v in (rank,iterations,oversamples,seed)):
        raise ValueError('Python integer controls required')
    if not 0<rank<min(x.shape) or iterations<1 or oversamples<1 or not 0<=seed<2**32:
        raise ValueError('Invalid projection controls')
    transpose=x.shape[0]<x.shape[1]
    oriented=x.T if transpose else x
    with threadpool_limits(limits=1):
        basis=randomized_range_finder(oriented,size=rank+oversamples,n_iter=iterations,
                                      power_iteration_normalizer='auto',random_state=seed)
        projected=basis.T@oriented
        core,singular,right=linalg.svd(projected,full_matrices=False,lapack_driver='gesdd')
        lifted=np.einsum('ik,kj->ij',basis,core,optimize=False)
        left,components=(right[:rank].T,lifted[:,:rank].T) if transpose else (lifted[:,:rank],right[:rank])
        _,components=svd_flip(left,components,u_based_decision=False)
    if not np.isfinite(components).all() or not np.isfinite(singular).all():
        raise ValueError('Nonfinite projection factors')
    return components.copy(),singular[:rank].copy()


def transform_projection(matrix, components):
    """Project finite float64 CSR rows through a supplied finite component matrix."""
    x=_matrix(matrix)
    if not isinstance(components,np.ndarray) or components.dtype!=np.float64 or components.ndim!=2:
        raise ValueError('Float64 component matrix required')
    if not x.shape[0] or not components.shape[0] or components.shape[1]!=x.shape[1] or not np.isfinite(components).all():
        raise ValueError('Nonempty aligned finite factors required')
    with threadpool_limits(limits=1):result=x@components.T
    if not np.isfinite(result).all():raise ValueError('Nonfinite projection')
    return np.asarray(result).copy()
