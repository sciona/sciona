"""Constructed domains for complete zero-shot workflow validation.

These are geometric symbols, not real glyphs, fonts or dataset excerpts. The
third joint class exists in both font populations and held-out populations,
but never in handwriting training. Separate draws prevent pixel-level reuse
between training, calibration and query populations.
"""
import numpy as np

from sciona.bengali_population import ImagePopulation


KNOWN = (0, 96)
UNSEEN = 192


def _population(labels, *, style, seed):
    rng = np.random.default_rng(seed)
    y, x = np.indices((137, 236))
    images = []
    for label in labels:
        dx, dy = rng.integers(-4, 5, size=2)
        vertical = (np.abs(x-118-dx) <= 18) & (np.abs(y-68-dy) <= 45)
        horizontal = (np.abs(x-118-dx) <= 75) & (np.abs(y-68-dy) <= 15)
        mask = vertical if label == KNOWN[0] else horizontal if label == KNOWN[1] else vertical | horizontal
        if style == 'outline':
            inner_vertical = (np.abs(x-118-dx) < 10) & (np.abs(y-68-dy) < 37)
            inner_horizontal = (np.abs(x-118-dx) < 67) & (np.abs(y-68-dy) < 7)
            inner = inner_vertical if label == KNOWN[0] else inner_horizontal if label == KNOWN[1] else inner_vertical | inner_horizontal
            mask &= ~inner
        image = rng.integers(0, 24, (137, 236), dtype=np.uint8)
        image[mask] += 200 if style != 'hand' else 175
        images.append(image)
    return ImagePopulation(np.stack(images), np.asarray(labels, dtype=np.int64))


def complete_populations():
    known_mask = np.zeros(14784, dtype=bool)
    known_mask[list(KNOWN)] = True
    known_mask.flags.writeable = False
    hand = _population(KNOWN*2, style='hand', seed=3101)
    fonts = tuple(_population((KNOWN+(UNSEEN,))*2, style=style, seed=seed)
                  for style, seed in [('solid', 3102), ('outline', 3103)])
    return dict(hand=hand, fonts=fonts,
                calibration=_population(KNOWN+(UNSEEN,), style='hand', seed=3104),
                query=_population(KNOWN+(UNSEEN,), style='hand', seed=3105),
                known_class_mask=known_mask,
                ood_training_labels=np.searchsorted(np.asarray(KNOWN), hand.labels).astype(np.int64))
