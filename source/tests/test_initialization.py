"""Independent M4 checks; no training experiments or reference demos are run.

Fixed seeds and sample-size-aware six-standard-error bounds avoid brittle
statistical assertions. Numerical measurements are attached to pytest JUnit.
"""
import json
import math

import pytest
import torch
from torch import nn

from source.activations import CustomActivation
from source.initialization import initialize_weights

METHODS = ['normal', 'fan_in_normal', 'xavier_normal', 'he_normal']
SIZES = [(512, 512), (512, 128), (128, 512), (12288, 128), (128, 64), (64, 1)]


@pytest.fixture(params=[torch.float32, torch.float64], ids=['float32', 'float64'])
def dtype(request):
    return request.param


@pytest.fixture(params=['cpu'] + (['cuda'] if torch.cuda.is_available() else []))
def device(request):
    return torch.device('cuda', torch.cuda.current_device()) if request.param == 'cuda' else torch.device('cpu')


@pytest.fixture(autouse=True)
def isolated_rng():
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(42)
        yield


@pytest.fixture(scope='module', autouse=True)
def bounded_threads():
    # Small test matrices do not benefit from oversubscribed CPU thread pools.
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 2))
    yield
    torch.set_num_threads(previous)


def expected_std(method, fan_in, fan_out):
    # Independent mathematical specification, not production helper calls.
    variance = {'normal': 1., 'fan_in_normal': 1. / fan_in,
                'xavier_normal': 2. / (fan_in + fan_out), 'he_normal': 2. / fan_in}
    return math.sqrt(variance[method])


def measurement(request, name, value):
    request.node.user_properties.append((name, json.dumps(value)))


class TestDistribution:
    @pytest.mark.parametrize('method', METHODS)
    @pytest.mark.parametrize('fan_in,fan_out', SIZES)
    def test_mean_std_and_weight_orientation(self, method, fan_in, fan_out, dtype, device, request):
        layer = nn.Linear(fan_in, fan_out, device=device, dtype=dtype)
        torch.manual_seed(42)
        initialize_weights(layer, method)
        w = layer.weight.detach().double()
        std = expected_std(method, fan_in, fan_out)
        n = w.numel()
        mean_tolerance = 6 * std / math.sqrt(n)
        std_tolerance = 6 * std / math.sqrt(2 * (n - 1))
        actual_mean, actual_std = w.mean().item(), w.std().item()
        measurement(request, 'distribution', dict(method=method, fan_in=fan_in, fan_out=fan_out,
                    dtype=str(dtype), device=str(device), n=n, expected_mean=0, expected_std=std,
                    expected_variance=std**2, actual_mean=actual_mean, actual_std=actual_std,
                    mean_tolerance=mean_tolerance, std_tolerance=std_tolerance))
        assert w.shape == (fan_out, fan_in)
        assert torch.isfinite(w).all()
        assert abs(actual_mean) <= mean_tolerance
        assert abs(actual_std - std) <= std_tolerance

    @pytest.mark.parametrize('fan_in,fan_out', [(512,128), (128,512), (64,1)])
    def test_scaling_for_same_random_draws(self, fan_in, fan_out, dtype, device):
        layer = nn.Linear(fan_in, fan_out, device=device, dtype=dtype)
        torch.manual_seed(42)
        initialize_weights(layer, 'normal')
        standard = layer.weight.detach().clone()
        for method in METHODS[1:]:
            torch.manual_seed(42)
            initialize_weights(layer, method)
            torch.testing.assert_close(layer.weight, standard * expected_std(method, fan_in, fan_out))


class TestBias:
    @pytest.mark.parametrize('method', METHODS)
    @pytest.mark.parametrize('bias', [True, False])
    def test_zero_bias_identity_or_no_bias(self, method, bias, dtype, device):
        layer = nn.Linear(8, 5, bias=bias, dtype=dtype, device=device)
        original = layer.bias
        if bias:
            with torch.no_grad():
                layer.bias.fill_(3)
        assert initialize_weights(layer, method) is None
        assert layer.bias is original
        if bias:
            assert torch.count_nonzero(layer.bias) == 0


class TestPreservation:
    @pytest.mark.parametrize('method', METHODS)
    @pytest.mark.parametrize('requires_grad', [True, False])
    def test_in_place_parameter_storage_metadata_and_graph(self, method, requires_grad, dtype, device):
        layer = nn.Linear(9, 4, dtype=dtype, device=device).requires_grad_(requires_grad)
        before = [(p, p.data_ptr(), p.shape, p.dtype, p.device, p.requires_grad) for p in layer.parameters()]
        with torch.enable_grad():
            initialize_weights(layer, method)
        for new, (old, pointer, shape, dt, dev, req) in zip(layer.parameters(), before):
            assert new is old and new.data_ptr() == pointer
            assert (new.shape, new.dtype, new.device, new.requires_grad) == (shape, dt, dev, req)
            assert new.is_leaf and new.grad_fn is None and new.grad is None


class TestReproducibility:
    @pytest.mark.parametrize('method', METHODS)
    def test_seed_repeat_change_and_sequential_layers(self, method, dtype, device, monkeypatch):
        first = nn.Linear(32, 16, dtype=dtype, device=device)
        second = nn.Linear(32, 16, dtype=dtype, device=device)
        samples = []
        for seed in (42, 42, 43):
            torch.manual_seed(seed)
            initialize_weights(first, method)
            samples.append(first.weight.detach().clone())
        assert torch.equal(samples[0], samples[1])
        assert not torch.equal(samples[0], samples[2])
        torch.manual_seed(42)
        def forbidden_seed(*args, **kwargs):
            pytest.fail('Initializer must not reset the RNG seed')
        monkeypatch.setattr(torch, 'manual_seed', forbidden_seed)
        initialize_weights(first, method)
        initialize_weights(second, method)
        assert torch.equal(first.weight, samples[0])
        assert not torch.equal(first.weight, second.weight)


class TestErrors:
    @pytest.mark.parametrize('method', ['unsupported', '', 'He_Normal', None, 42])
    @pytest.mark.parametrize('linear', [True, False])
    def test_invalid_method_raises_before_mutation(self, method, linear):
        module = nn.Linear(3, 2) if linear else nn.Conv2d(1, 2, 3)
        before = {k: v.clone() for k, v in module.state_dict().items()}
        with pytest.raises(ValueError, match='Unsupported initialization method'):
            initialize_weights(module, method)
        for key, value in module.state_dict().items():
            assert torch.equal(value, before[key])

    @pytest.mark.parametrize('method', METHODS)
    def test_non_linear_noop_and_no_implicit_recursion(self, method, dtype, device):
        module = nn.Sequential(nn.Conv2d(1, 2, 3), nn.BatchNorm2d(2), nn.Linear(4, 2)).to(device=device, dtype=dtype)
        before = {k: v.clone() for k, v in module.state_dict().items()}
        parameters = list(module.parameters())
        rng = torch.random.get_rng_state().clone()
        assert initialize_weights(module, method) is None
        assert torch.equal(rng, torch.random.get_rng_state())
        assert all(a is b for a, b in zip(parameters, module.parameters()))
        for key, value in module.state_dict().items():
            assert torch.equal(value, before[key])
        for child in list(module.children())[:2]:
            initialize_weights(child, method)
        for key, value in module.state_dict().items():
            assert torch.equal(value, before[key])

    def test_default_is_he_normal(self, dtype, device):
        layer = nn.Linear(16, 8, dtype=dtype, device=device)
        torch.manual_seed(42)
        initialize_weights(layer)
        default = layer.weight.detach().clone()
        torch.manual_seed(42)
        initialize_weights(layer, 'he_normal')
        assert torch.equal(default, layer.weight)


class TestSignalPropagation:
    def test_ten_layers_same_input_seed_and_sampling(self, dtype, device, request):
        # Match the notebook protocol: input seed 42, then reseed each method.
        torch.manual_seed(42)
        x = torch.randn(256, 512, dtype=dtype, device=device)
        original = x.clone()
        moments, weights = {}, {}
        with torch.no_grad():
            for method in METHODS:
                torch.manual_seed(42)
                h = x.clone()
                moments[method] = [h.square().mean().item()]
                weights[method] = []
                activation = CustomActivation('relu')
                for _ in range(10):
                    layer = nn.Linear(512, 512, dtype=dtype, device=device)
                    initialize_weights(layer, method)
                    weights[method].append(layer.weight.detach().clone())
                    h = activation(layer(h))
                    assert torch.isfinite(h).all()
                    moments[method].append(h.square().mean().item())
        max_weight_diff = max((a-b).abs().max().item() for a,b in zip(weights['fan_in_normal'], weights['xavier_normal']))
        max_moment_diff = max(abs(a-b) for a,b in zip(moments['fan_in_normal'], moments['xavier_normal']))
        measurement(request, 'signal', dict(dtype=str(dtype), device=str(device), moments=moments,
                    fan_xavier_max_weight_difference=max_weight_diff,
                    fan_xavier_max_moment_difference=max_moment_diff))
        assert torch.equal(x, original)
        assert max_weight_diff == 0 and max_moment_diff == 0
        # Broad qualitative bounds, declared before executing the tests.
        assert moments['normal'][-1] / moments['normal'][0] > 1e12
        assert moments['fan_in_normal'][-1] / moments['fan_in_normal'][0] < .02
        assert all(.25 < m / moments['he_normal'][0] < 4 for m in moments['he_normal'][1:])

    @pytest.mark.parametrize('fan_in,fan_out', [(512,128), (128,512)])
    def test_rectangular_fan_in_and_xavier_differ(self, fan_in, fan_out, dtype, device, request):
        x = torch.randn(64, fan_in, dtype=dtype, device=device)
        layer = nn.Linear(fan_in, fan_out, dtype=dtype, device=device)
        torch.manual_seed(42)
        initialize_weights(layer, 'fan_in_normal')
        w = layer.weight.detach().clone()
        with torch.no_grad():
            first = CustomActivation('relu')(layer(x))
        torch.manual_seed(42)
        initialize_weights(layer, 'xavier_normal')
        ratio = math.sqrt(2 * fan_in / (fan_in + fan_out))
        torch.testing.assert_close(layer.weight, w * ratio)
        assert not torch.equal(w, layer.weight)
        with torch.no_grad():
            second = CustomActivation('relu')(layer(x))
        torch.testing.assert_close(second.square().mean(), first.square().mean() * ratio**2)
        measurement(request, 'rectangular_signal', dict(fan_in=fan_in, fan_out=fan_out, dtype=str(dtype),
                    expected_moment_ratio=ratio**2, actual_moment_ratio=(second.square().mean()/first.square().mean()).item()))


class TestIntegration:
    @pytest.mark.parametrize('method', METHODS)
    def test_apply_custom_relu_backward_and_existing_optimizer(self, method, dtype, device):
        net = nn.Sequential(nn.Linear(8, 16), CustomActivation('relu'), nn.Linear(16, 2)).to(device=device, dtype=dtype)
        optimizer = torch.optim.SGD(net.parameters(), lr=.01)
        params = list(net.parameters())
        assert net.apply(lambda m: initialize_weights(m, method)) is net
        assert all(a is b for a,b in zip(params, net.parameters()))
        assert all(a is b for a,b in zip(params, optimizer.param_groups[0]['params']))
        x = torch.linspace(-1, 1, 64, dtype=dtype, device=device).reshape(8, 8).requires_grad_()
        y = net(x)
        assert y.shape == (8, 2) and torch.isfinite(y).all()
        y.square().mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        before = [p.detach().clone() for p in params]
        grads = []
        for p in params:
            assert p.grad is not None and torch.isfinite(p.grad).all()
            assert torch.count_nonzero(p.grad) > 0
            grads.append(p.grad.clone())
        optimizer.step()
        for p, old, grad in zip(params, before, grads):
            torch.testing.assert_close(p, old - .01 * grad)
            assert not torch.equal(p, old)
