"""Reuse vector selection while retaining cell confidence weighting in the recipe."""
import numpy as np
from sciona.decomposed_trackml_lifecycle import _bound
from sciona.decomposed_trackml_tangent_lifecycle import execute as tangent_execute
from sciona.decomposed_trackml_residual_lifecycle import residual_class


def execute(*args, vector_selection_function, **kwargs):
    if not callable(vector_selection_function):
        raise ValueError('Callable vector selection provider required')

    class CellDirections:
        def __init__(self, retained):
            self.retained = retained

        def __getattr__(self,name):
            return getattr(self.retained,name)

        def estimateClosestDirection(self,hit_id,udir):
            first,second,count = self.retained.cellFeaturesByHitId(hit_id)
            selected,inner = vector_selection_function(udir,first,second)
            if (selected.dtype != np.float64 or selected.shape != udir.shape
                    or inner.dtype != np.float64 or inner.shape != (len(udir),)
                    or not np.isfinite(selected).all() or not np.isfinite(inner).all()
                    or np.any(inner < 0)):
                raise ValueError('Finite aligned vector selection outputs required')
            # Empirical cell-count confidence remains source recipe policy.
            return selected,inner,np.sqrt(count)*(1-inner)

    def local_class(base, displacement_function, uncertainty_function):
        parent = residual_class(base,displacement_function,uncertainty_function)
        class CellSelection(parent):
            def setupRun(self,*values,**options):
                run = super().setupRun(*values,**options)
                if run.cell_features is not None and not isinstance(run.cell_features,CellDirections):
                    run.cell_features = CellDirections(run.cell_features)
                return run
        return CellSelection
    return _bound(tangent_execute, {'residual_class': local_class})(*args,**kwargs)
