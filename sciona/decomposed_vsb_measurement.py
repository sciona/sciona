"""Explicit measurement policy over externally supplied numeric operations."""
import numpy as np


def collect(values,coefficients,controls,arguments_function,filter_function,residual_function,
            maxima_function,rank_function,gradient_function,selection_function,
            descriptor_function,phase_function):
    """Collect ordered peak records, leaving aggregate computation downstream.

    Columns form consecutive triples. Each supplied coefficient corresponds to
    that same column and full signal length. Shape checks do not authenticate
    this provenance or callable semantics; qualified graph bindings must do so.
    """
    x=np.asarray(values);coefficients=np.asarray(coefficients)
    if x.ndim!=2 or len(x)<3 or not x.shape[1] or x.shape[1]%3 or x.dtype.kind not in 'iuf' or not np.isfinite(x).all():
        raise ValueError('Finite sample-by-signal matrix with complete triples required')
    if coefficients.shape!=(x.shape[1],) or coefficients.dtype.kind not in 'iufc' or not np.isfinite(coefficients).all():
        raise ValueError('Aligned finite Fourier coefficients required')
    expected={'retention','window','width','trim','threshold','count','offset','small_radius','large_radius'}
    if type(controls) is not dict or set(controls)!=expected:raise ValueError('Explicit feature controls required')
    functions=(arguments_function,filter_function,residual_function,maxima_function,rank_function,
               gradient_function,selection_function,descriptor_function,phase_function)
    if not all(callable(fn) for fn in functions):raise ValueError('Explicit numeric callables required')
    groups=[];heights=[];descriptors=[];quadrants=[]
    for column in range(x.shape[1]):
        b,a,signal,axis,zi=arguments_function(x[:,column],controls['retention'])
        residual=residual_function(filter_function(b,a,signal,axis=axis,zi=zi))
        candidates=maxima_function(np.abs(residual),window=controls['window'])
        ranked,amplitudes=rank_function(residual,candidates)
        gradient=gradient_function(amplitudes,controls['width'],controls['trim'])
        positions,amplitudes,_=selection_function(ranked,amplitudes,gradient,controls['threshold'],controls['count'],controls['offset'])
        features=descriptor_function(positions,residual,small_radius=controls['small_radius'],large_radius=controls['large_radius'])
        phases=phase_function(positions,coefficients[column],length=len(x))
        if features.shape!=(len(positions),4) or phases.shape!=positions.shape or amplitudes.shape!=positions.shape:
            raise ValueError('Aligned peak record outputs required')
        groups.extend([column//3]*len(positions));heights.extend(amplitudes)
        descriptors.extend(features);quadrants.extend(phases)
    return (np.asarray(groups,dtype=int),np.asarray(heights,dtype=float),
            np.asarray(descriptors,dtype=float).reshape(-1,4),np.asarray(quadrants,dtype=int),x.shape[1]//3)
