"""Torch 1.4 Adam arithmetic order for the winner's dense detector updates.

Only the source configuration is exposed: betas .9/.999, epsilon 1e-8,
no weight decay and no AMSGrad. Apex precision/scaling is a separate layer.
"""
import math

import torch
from torch.optim import Optimizer


class LegacyDetectionAdam(Optimizer):
    def __init__(self, params, lr):
        if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr < 0:
            raise ValueError('finite nonnegative learning rate required')
        super().__init__(params, dict(lr=lr, betas=(.9, .999), eps=1e-8,
                                     weight_decay=0, amsgrad=False))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if (group['betas'] != (.9, .999) or group['eps'] != 1e-8
                    or group['weight_decay'] != 0 or group['amsgrad']):
                raise ValueError('source detector Adam configuration required')
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.layout != torch.strided or parameter.dtype not in (torch.float32, torch.float64):
                    raise ValueError('dense float32 or float64 detector gradients required')
                state = self.state[parameter]
                if not state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                state['step'] += 1
                first, second = state['exp_avg'], state['exp_avg_sq']
                first.mul_(.9).add_(gradient, alpha=1-.9)
                second.mul_(.999).addcmul_(gradient, gradient, value=1-.999)
                denominator = second.sqrt().div_(math.sqrt(1-.999**state['step'])).add_(1e-8)
                parameter.addcdiv_(first, denominator, value=-group['lr']/(1-.9**state['step']))
        return loss
