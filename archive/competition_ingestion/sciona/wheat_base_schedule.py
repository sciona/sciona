"""Learning rates produced by the source's one-epoch warmup composition."""
import math


def base_learning_rate(epoch, detector):
    if type(epoch) is not int or not 0 <= epoch < 100:
        raise ValueError('source epoch must be in 0..99')
    if detector not in ('effdet', 'fasterrcnn'):
        raise ValueError('source detector family required')
    requested = .0005 if detector == 'effdet' else .005
    initial = requested / 10
    if epoch in (0, 2):
        return initial
    if epoch == 1:
        return initial * 10
    return (initial * 10) * (1 + math.cos(math.pi * (epoch - 1) / 99)) / 2
