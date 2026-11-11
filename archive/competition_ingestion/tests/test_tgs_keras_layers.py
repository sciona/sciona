import torch

from sciona.tgs_keras_layers import KerasBatchNorm, KerasDecoder, DecoderAttention


def test_population_normalization_and_historical_double_variance_correction():
    layer = KerasBatchNorm(1, epsilon=.001, retention=.9)
    values = torch.tensor([1., 3.]).reshape(2, 1, 1, 1).requires_grad_()
    output = layer(values)
    torch.testing.assert_close(output.flatten(), torch.tensor([-1., 1.]) / (1.001 ** .5))
    torch.testing.assert_close(layer.moving_mean, torch.tensor([.2]))
    # TensorFlow's returned variance is 2; Keras2.2.0 then multiplies by
    # 2/(2-(1+epsilon)) before updating the moving statistic.
    expected_variance = .9 + .1 * 2 * (2 / .999)
    torch.testing.assert_close(layer.moving_variance, torch.tensor([expected_variance]))
    layer.eval()
    torch.testing.assert_close(layer(values), (values - .2) / ((expected_variance + .001) ** .5))


def test_no_scale_and_singleton_variance_are_defined():
    layer = KerasBatchNorm(2, scale=False)
    assert 'gamma' not in dict(layer.named_parameters())
    output = layer(torch.tensor([3., 4.]).reshape(1, 2, 1, 1))
    assert not output.any()
    torch.testing.assert_close(layer.moving_variance, torch.full((2,), .99))


def test_attention_has_channelwise_spatial_gates():
    layer = DecoderAttention(8)
    assert layer.local_gate.out_channels == 8
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.zero_()
    value = torch.randn(2, 8, 3, 3)
    torch.testing.assert_close(layer(value), value)


def test_decoder_shape_and_both_input_gradients():
    layer = KerasDecoder(16, 8, 4)
    value = torch.rand(2, 16, 4, 4, requires_grad=True)
    skip = torch.rand(2, 8, 8, 8, requires_grad=True)
    output = layer(value, skip)
    assert output.shape == (2, 4, 8, 8)
    output.square().mean().backward()
    assert value.grad.abs().sum() > 0 and skip.grad.abs().sum() > 0
