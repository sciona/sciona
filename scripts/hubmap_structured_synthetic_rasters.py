"""Constructed spatial foreground cue; no real images or dataset metadata."""
import numpy as np


def labeled_raster(seed,foreground,side=1024):
    if type(foreground) is not bool or type(side) is not int or side<1:
        raise ValueError('Boolean foreground and positive integer side required')
    rng=np.random.default_rng(seed)
    y,x=np.ogrid[:side,:side]
    cy,cx=rng.uniform(.4,.6,2)*side
    radius=rng.uniform(.25,.33)*side
    mask=(((y-cy)**2+(x-cx)**2<radius**2)&foreground).astype(np.uint8)
    image=rng.integers(0,66,(side,side,3),dtype=np.uint8)
    image+=np.where(mask[...,None],190,15).astype(np.uint8)
    return image,mask
