"""Synthetic raster fixture with a learnable, explicit foreground intensity cue.

No real image, mask, identifier or dataset metadata is used. The masks remain
binary and the RGB inputs retain within-class variation and source tile padding.
"""
import numpy as np


def labeled_raster(seed,foreground,side=600):
    if type(foreground) is not bool or type(side) is not int or side<1:
        raise ValueError('Boolean foreground and positive integer side required')
    low,high=(190,256) if foreground else (15,81)
    image=np.random.default_rng(seed).integers(low,high,(side,side,3),dtype=np.uint8)
    mask=np.full((side,side),int(foreground),dtype=np.uint8)
    return image,mask
