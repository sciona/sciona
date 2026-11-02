import numpy as np
import pytest
from sciona import dsb_decomposed_annotations as candidate
from sciona.dsb_training_preprocessing import training_voxel_annotations
from sciona.dsb_world_preprocessing import world_training_annotations


@pytest.mark.parametrize('count',[0,1,7])
@pytest.mark.parametrize('flip',[False,True])
def test_both_annotation_paths_preserve_source_policy(count,flip):
    rows=np.arange(count*4,dtype=float).reshape(count,4)+1
    spacing=np.array([.7,1.3,2.1]);bounds=np.array([[2,9],[4,11],[1,8]])
    resolution=(.6,1.1,1.7)
    np.testing.assert_array_equal(candidate.training_voxel_annotations(rows,spacing,bounds,resolution),
                                  training_voxel_annotations(rows,spacing,bounds,resolution))
    args=(rows,[10.,20.,30.],spacing,bounds,(16,20,24))
    np.testing.assert_array_equal(candidate.world_training_annotations(*args,flip=flip,requested_spacing=resolution),
                                  world_training_annotations(*args,flip=flip,requested_spacing=resolution))
