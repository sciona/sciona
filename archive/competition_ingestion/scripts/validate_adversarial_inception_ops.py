"""Synthetic scalar value/gradient oracles for Inception boundary operations."""
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sciona.adversarial_inception_ops import InceptionConv, pool2d


def geometry(size, kernel, stride, padding):
    # Independently enumerate output origins according to TF SAME/VALID contract.
    count = len(range(0, size, stride)) if padding == 'SAME' else len(range(0, size-kernel+1, stride))
    before = max((count-1)*stride+kernel-size, 0)//2 if padding == 'SAME' else 0
    return [i*stride-before for i in range(count)]


def pool_oracle(a, kernel, stride, padding, mode):
    ys = geometry(a.shape[2], kernel[0], stride[0], padding)
    xs = geometry(a.shape[3], kernel[1], stride[1], padding)
    out = np.zeros((*a.shape[:2], len(ys), len(xs)))
    dx = np.zeros_like(a)
    for n, c, i, j in itertools.product(range(a.shape[0]), range(a.shape[1]), range(len(ys)), range(len(xs))):
        points = [(y, x) for y in range(ys[i], ys[i]+kernel[0]) for x in range(xs[j], xs[j]+kernel[1])
                  if 0 <= y < a.shape[2] and 0 <= x < a.shape[3]]
        values = [a[n, c, y, x] for y, x in points]
        if mode == 'avg':
            out[n, c, i, j] = sum(values)/len(values)
            for y, x in points: dx[n, c, y, x] += 1/len(values)
        else:
            selected = int(np.argmax(values)); out[n, c, i, j] = values[selected]
            y, x = points[selected]; dx[n, c, y, x] += 1
    return out, dx


def conv_oracle(a, w, b, stride, padding):
    ys = geometry(a.shape[2], w.shape[2], stride[0], padding)
    xs = geometry(a.shape[3], w.shape[3], stride[1], padding)
    out = np.zeros((a.shape[0], w.shape[0], len(ys), len(xs)))
    dx, dw = np.zeros_like(a), np.zeros_like(w)
    for n, o, i, j in itertools.product(range(a.shape[0]), range(w.shape[0]), range(len(ys)), range(len(xs))):
        out[n, o, i, j] = b[o]
        for c, u, v in itertools.product(range(a.shape[1]), range(w.shape[2]), range(w.shape[3])):
            y, x = ys[i]+u, xs[j]+v
            if 0 <= y < a.shape[2] and 0 <= x < a.shape[3]:
                out[n, o, i, j] += a[n, c, y, x]*w[o, c, u, v]
                dx[n, c, y, x] += w[o, c, u, v]
                dw[o, c, u, v] += a[n, c, y, x]
    return out, dx, dw, np.full_like(b, a.shape[0]*len(ys)*len(xs))


def main():
    torch.set_num_threads(2)
    reference = ROOT/'docs/reviews/competition_adversarial_pooling_reference.json'
    pin = json.loads(reference.read_text())
    for item in pin['files']:
        p = Path('/private/tmp/sciona_adversarial_tensorflow_source')/item['path']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == item['sha256']
    # Pin the examined historical exclusion/counting rule; C++ is not executed here.
    source = (Path('/private/tmp/sciona_adversarial_tensorflow_source')/'tensorflow/core/kernels/eigen_pooling.h').read_text()
    assert 'if (t != -Eigen::NumTraits<T>::highest())' in source
    assert 'return accum / T(scalarCount_);' in source
    rng = np.random.default_rng(712)
    pooling = convolution = 0
    for dtype, shape, kernel, stride, padding, mode in itertools.product(
            (torch.float32, torch.float64), ((5, 7), (6, 8)), ((3, 3), (2, 4), (1, 7)),
            ((1, 1), (2, 2)), ('SAME', 'VALID'), ('avg', 'max')):
        a = -rng.uniform(.1, 5, (2, 2, *shape))
        x = torch.tensor(a, dtype=dtype, requires_grad=True)
        expected, dx = pool_oracle(x.detach().numpy(), kernel, stride, padding, mode)
        out = pool2d(x, kernel, stride=stride, padding=padding, mode=mode)
        out.sum().backward()
        tol = 2e-6 if dtype == torch.float32 else 1e-12
        np.testing.assert_allclose(out.detach().numpy(), expected, rtol=tol, atol=tol)
        np.testing.assert_allclose(x.grad.numpy(), dx, rtol=tol, atol=tol)
        pooling += 1
    for shape, kernel, stride, padding in itertools.product(
            ((5, 7), (6, 8)), ((3, 3), (1, 7), (7, 1), (2, 4)), ((1, 1), (2, 2)), ('SAME', 'VALID')):
        if padding == 'VALID' and any(k > n for k, n in zip(kernel, shape)): continue
        layer = InceptionConv(2, 3, kernel, stride=stride, padding=padding, normalized=False, activation=False).double()
        a = rng.normal(size=(2, 2, *shape)); w = rng.normal(size=tuple(layer.conv.weight.shape)); b = rng.normal(size=3)
        with torch.no_grad():
            layer.conv.weight.copy_(torch.tensor(w)); layer.conv.bias.copy_(torch.tensor(b))
        x = torch.tensor(a, requires_grad=True)
        out = layer(x); out.sum().backward()
        expected, dx, dw, db = conv_oracle(a, w, b, stride, padding)
        for actual, target in [(out.detach().numpy(), expected), (x.grad.numpy(), dx),
                               (layer.conv.weight.grad.numpy(), dw), (layer.conv.bias.grad.numpy(), db)]:
            np.testing.assert_allclose(actual, target, rtol=1e-12, atol=1e-12)
        convolution += 1
    # Independent 1x1 affine/normalization/ReLU oracle with nonidentity stored state.
    layer = InceptionConv(1, 2, 1).double()
    with torch.no_grad():
        layer.conv.weight.copy_(torch.tensor([2., -3.]).reshape(2, 1, 1, 1))
        layer.norm.moving_mean.copy_(torch.tensor([.3, -.2]))
        layer.norm.moving_variance.copy_(torch.tensor([.5, 2.]))
        layer.norm.beta.copy_(torch.tensor([.7, -.4]))
    x = torch.tensor([[[[-2., -.1, .4, 3.]]]], requires_grad=True)
    x = x.detach().double().requires_grad_()
    buffers = {k: v.clone() for k, v in layer.named_buffers()}
    factor = (layer.norm.moving_variance.numpy()+.001)**-.5
    affine = (x.detach().numpy()*np.array([2., -3.])[None, :, None, None]-layer.norm.moving_mean.numpy()[None, :, None, None])*factor[None, :, None, None]+layer.norm.beta.numpy()[None, :, None, None]
    out = layer(x); out.sum().backward()
    np.testing.assert_allclose(out.detach().numpy(), np.maximum(affine, 0), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(x.grad.numpy(), ((affine > 0)* (factor*np.array([2., -3.]))[None, :, None, None]).sum(axis=1, keepdims=True), rtol=1e-12, atol=1e-12)
    assert layer.conv.bias is None and layer.norm.gamma is None
    assert torch.equal(layer.train()(x), layer.eval()(x))
    assert all(torch.equal(v, buffers[k]) for k, v in layer.named_buffers())
    rejected = 0
    for call in [lambda: pool2d(torch.ones(1, 1, 2, 2), 3, padding='VALID'),
                 lambda: pool2d(torch.full((1, 1, 2, 2), float('nan')), 1),
                 lambda: pool2d(torch.full((1, 1, 2, 2), torch.finfo(torch.float32).min), 1),
                 lambda: InceptionConv(1, 2, 3, stride=0),
                 lambda: InceptionConv(1, 2, 3, padding='same')]:
        try: call()
        except ValueError: rejected += 1
        else: raise AssertionError('invalid boundary accepted')
    paths = ['sciona/adversarial_inception_ops.py', 'sciona/adversarial_normalization.py',
             'scripts/validate_adversarial_inception_ops.py', 'docs/reviews/competition_adversarial_pooling_reference.json',
             'docs/licenses/Adversarial-TensorFlow-reference-Apache-2.0.txt']
    report = {'format': 'adversarial-inception-ops-validation.v1', 'result': 'passed',
              'checks': {'scalar_pool_output_input_gradient_cases': pooling,
                         'scalar_conv_output_input_weight_bias_gradient_cases': convolution,
                         'independent_normalized_relu_values_input_gradients': True,
                         'frozen_state_and_mode_invariance': True, 'invalid_cases_rejected': rejected},
              'limits': 'Synthetic CPU float32/64 pooling and float64 convolution. Historical source reference inspected and hashed, not compiled or binary-compared. Max-pool gradient fixtures have unique maxima; tie parity unproven. Full Inception topology, heads, checkpoints and attack integration remain unvalidated. Historical average-pool sentinel collision explicitly rejected.',
              'sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}}
    (ROOT/'docs/reviews/competition_adversarial_inception_ops.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__ == '__main__':
    main()
