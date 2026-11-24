"""Independent M5 architecture/integration checks, not full-dataset training."""
import json
import random
from unittest.mock import Mock

import pytest
import torch
from torch import nn

import source.model as models
import source.activations as acts
from source.config import CONFIG
from source.dataset import build_loaders
from source.augmentation import build_augmentation
from source.preprocessing import build_preprocessing

METHODS = ['normal', 'fan_in_normal', 'xavier_normal', 'he_normal']


@pytest.fixture(params=[torch.float32, torch.float64], ids=['float32','float64'])
def dtype(request):
    return request.param


@pytest.fixture(params=['cpu'] + (['cuda'] if torch.cuda.is_available() else []))
def device(request):
    return torch.device('cuda', torch.cuda.current_device()) if request.param == 'cuda' else torch.device('cpu')


@pytest.fixture(autouse=True)
def isolated_rng():
    state = random.getstate()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(42)
        random.seed(42)
        yield
    random.setstate(state)


@pytest.fixture(scope='module', autouse=True)
def bounded_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(min(old, 2))
    yield
    torch.set_num_threads(old)


@pytest.fixture(scope='module')
def fixed_real_batch():
    state = random.getstate()
    with torch.random.fork_rng():
        random.seed(42)
        torch.manual_seed(42)
        loaders = build_loaders(CONFIG, train_transform=build_augmentation(CONFIG))
        images, targets = next(iter(loaders['train']))
    random.setstate(state)
    return images, targets


def record(request, name, value):
    request.node.user_properties.append((name, json.dumps(value)))


def make_model(dtype, device, activation='relu'):
    model = models.MLP(activation=activation).to(device=device, dtype=dtype)
    models.initialize_model(model)
    return model


def check_gradients(model):
    for p in model.parameters():
        assert p.grad is not None and p.grad.shape == p.shape
        assert torch.isfinite(p.grad).all()
        assert torch.count_nonzero(p.grad) > 0


class TestArchitecture:
    @pytest.mark.parametrize('activation', ['relu', 'sigmoid'])
    @pytest.mark.parametrize('hidden', [(128,64), (17,), [16,8,4], ()])
    def test_dimensions_order_and_registered_modules(self, activation, hidden):
        model = models.MLP(hidden_dims=hidden, activation=activation)
        layers = list(model.layers)
        assert len(layers) == 2 * len(hidden) + 1
        previous = 12288
        for i, width in enumerate(hidden):
            assert type(layers[2*i]) is nn.Linear
            assert (layers[2*i].in_features, layers[2*i].out_features) == (previous, width)
            assert type(layers[2*i+1]) is acts.CustomActivation
            assert layers[2*i+1].name == activation
            previous = width
        assert type(layers[-1]) is nn.Linear
        assert (layers[-1].in_features, layers[-1].out_features) == (previous,1)
        assert all(any(child is m for m in model.modules()) for child in layers)
        assert not any(isinstance(m, (nn.ReLU, nn.Sigmoid)) for m in model.modules())

    def test_output_is_unbounded_logit(self, dtype, device):
        model = make_model(dtype, device)
        with torch.no_grad():
            model.layers[-1].weight.zero_()
            for value in (-3., 3.):
                model.layers[-1].bias.fill_(value)
                assert torch.equal(model(torch.zeros(2,3,64,64, dtype=dtype, device=device)),
                                   torch.full((2,1),value,dtype=dtype,device=device))


class TestForward:
    @pytest.mark.parametrize('batch', [1,2,32])
    @pytest.mark.parametrize('activation', ['relu','sigmoid'])
    def test_batch_flatten_independence_and_noncontiguous(self, batch, activation, dtype, device):
        model = make_model(dtype, device, activation)
        x = torch.randn(batch,3,64,64,dtype=dtype,device=device).transpose(2,3)
        assert not x.is_contiguous()
        old = x.clone()
        with torch.no_grad():
            output = model(x)
            per_image = torch.cat([model(x[i:i+1]) for i in range(batch)])
            torch.testing.assert_close(output, per_image, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(output, model(x.contiguous()))
            torch.testing.assert_close(output, model.layers(x.reshape(batch,12288)))
            torch.testing.assert_close(model(x.flip(0)), output.flip(0))
        assert output.shape == (batch,1) and output.dtype == dtype and output.device == device
        assert torch.isfinite(output).all() and torch.equal(x,old)


class TestParameters:
    def test_count_registration_and_safe_state_dict_roundtrip(self, dtype, device, tmp_path, request):
        model = make_model(dtype, device)
        expected = {'layers.0.weight': (128,12288), 'layers.0.bias': (128,),
                    'layers.2.weight': (64,128), 'layers.2.bias': (64,),
                    'layers.4.weight': (1,64), 'layers.4.bias': (1,)}
        params = dict(model.named_parameters())
        assert set(params) == set(expected) == set(model.state_dict())
        assert sum(p.numel() for p in params.values()) == 1581313
        for name, shape in expected.items():
            assert params[name].shape == shape and params[name].requires_grad
            assert params[name].is_leaf and params[name].dtype == dtype
        record(request,'parameter_count',1581313)
        path = tmp_path / 'state.pt'
        torch.save(model.state_dict(),path)
        clone = models.MLP().to(device=device,dtype=dtype)
        clone.load_state_dict(torch.load(path,map_location=device,weights_only=True),strict=True)
        x = torch.randn(2,3,64,64,dtype=dtype,device=device)
        torch.testing.assert_close(model(x),clone(x),rtol=0,atol=0)


class TestInitialization:
    @pytest.mark.parametrize('method', METHODS)
    def test_hidden_dispatch_output_xavier_identity_and_seed(self, method, dtype, device, monkeypatch):
        model = models.MLP().to(device=device,dtype=dtype)
        params = list(model.parameters())
        pointers = [p.data_ptr() for p in params]
        spy = Mock(wraps=models.initialize_weights)
        monkeypatch.setattr(models, 'initialize_weights', spy)
        torch.manual_seed(42)
        assert models.initialize_model(model,method) is None
        linears = [model.layers[i] for i in (0,2,4)]
        assert spy.call_count == 3
        assert [call.args for call in spy.call_args_list] == [(linears[0],method),(linears[1],method),(linears[2],'xavier_normal')]
        first = [p.detach().clone() for p in params]
        assert all(a is b and b.data_ptr()==ptr for a,b,ptr in zip(params,model.parameters(),pointers))
        assert all(torch.count_nonzero(layer.bias)==0 for layer in linears)
        torch.manual_seed(42)
        models.initialize_model(model,method)
        assert all(torch.equal(a,b) for a,b in zip(first,params))
        torch.manual_seed(43)
        models.initialize_model(model,method)
        assert not torch.equal(first[0],params[0])
        def forbidden(*args, **kwargs):
            pytest.fail('initialize_model must not reset the seed')
        monkeypatch.setattr(torch,'manual_seed',forbidden)
        before = params[0].detach().clone()
        models.initialize_model(model,method)
        assert not torch.equal(before,params[0])


class TestCustomActivation:
    @pytest.mark.parametrize('activation',['relu','sigmoid'])
    def test_custom_backward_and_derivative_are_called(self, activation, dtype, device, monkeypatch):
        model = make_model(dtype,device,activation)
        backward = Mock(wraps=acts.CustomActivationFunction.backward)
        derivative = Mock(wraps=acts.activation_derivative)
        monkeypatch.setattr(acts.CustomActivationFunction,'backward',staticmethod(backward))
        monkeypatch.setattr(acts,'activation_derivative',derivative)
        model(torch.randn(2,3,64,64,dtype=dtype,device=device)).sum().backward()
        assert backward.call_count == derivative.call_count == 2
        assert [call.args[1] for call in derivative.call_args_list] == [activation,activation]
        assert [tuple(call.args[0].shape) for call in derivative.call_args_list] == [(2,64),(2,128)]


class TestOptimization:
    @pytest.mark.parametrize('activation',['relu','sigmoid'])
    def test_bce_gradients_and_adam_updates_all_parameters(self, activation, dtype, device):
        model = make_model(dtype,device,activation)
        optimizer = torch.optim.Adam(model.parameters(),lr=.001)
        x = torch.randn(8,3,64,64,dtype=dtype,device=device)
        y = torch.tensor([[0],[1]]*4,dtype=dtype,device=device)
        loss = nn.BCEWithLogitsLoss()(model(x),y)
        assert torch.isfinite(loss)
        loss.backward()
        check_gradients(model)
        old = [p.detach().clone() for p in model.parameters()]
        optimizer.step()
        assert all(not torch.equal(a,b) for a,b in zip(old,model.parameters()))

    def test_fixed_real_batch_learning_100_steps(self, dtype, device, fixed_real_batch, request):
        model = make_model(dtype,device)
        optimizer = torch.optim.Adam(model.parameters(),lr=.001)
        x,y = [t.to(device=device,dtype=dtype).clone() for t in fixed_real_batch]
        old_x,old_y = x.clone(),y.clone()
        criterion = nn.BCEWithLogitsLoss()
        history = []
        for _ in range(100):
            optimizer.zero_grad()
            loss = criterion(model(x),y)
            assert torch.isfinite(loss)
            history.append(loss.item())
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            final = criterion(model(x),y).item()
        record(request,'single_batch_learning',dict(dtype=str(dtype),device=str(device),steps=100,
               initial_loss=history[0],final_loss=final,history=history,
               required_final_initial_ratio=.1,seed=42,samples=32,source='fixed augmented training batch'))
        assert final < .1 * history[0]
        assert torch.equal(x,old_x) and torch.equal(y,old_y)


class TestEndToEnd:
    def test_real_loaders_train_only_augmentation_and_step(self, dtype, device):
        train = Mock(wraps=build_augmentation(CONFIG))
        evaluate = Mock(wraps=build_preprocessing(CONFIG.image_size))
        loaders = build_loaders(CONFIG,train_transform=train,eval_transform=evaluate)
        plain = build_loaders(CONFIG)
        images,targets = next(iter(loaders['train']))
        _,expected_targets = next(iter(plain['train']))
        assert torch.equal(targets,expected_targets)
        assert train.call_count==32 and evaluate.call_count==0
        for name in ('val','test'):
            a,b = next(iter(loaders[name]))
            c,d = next(iter(loaders[name]))
            assert torch.equal(a,c) and torch.equal(b,d)
        assert train.call_count==32 and evaluate.call_count==128
        assert images.shape==(32,3,64,64) and targets.shape==(32,1)
        model = make_model(dtype,device)
        optimizer = torch.optim.Adam(model.parameters(),lr=.001)
        logits = model(images.to(device=device,dtype=dtype))
        assert logits.shape==(32,1) and torch.isfinite(logits).all()
        loss = nn.BCEWithLogitsLoss()(logits,targets.to(device=device,dtype=dtype))
        assert torch.isfinite(loss)
        loss.backward()
        check_gradients(model)
        before = model.layers[0].weight.detach().clone()
        optimizer.step()
        assert not torch.equal(before,model.layers[0].weight)


class TestErrors:
    @pytest.mark.parametrize('name',['tanh','',None])
    def test_invalid_activation_with_hidden_layers(self,name):
        with pytest.raises(ValueError,match='Unsupported activation'):
            models.MLP(activation=name)

    @pytest.mark.parametrize('method',['invalid','',None])
    def test_invalid_initialization_default_architecture(self,method):
        with pytest.raises(ValueError,match='Unsupported initialization'):
            models.initialize_model(models.MLP(),method)

    @pytest.mark.parametrize('shape',[(2,3,32,32),(2,12287)])
    def test_incompatible_feature_count(self,shape,dtype,device):
        model = make_model(dtype,device)
        with pytest.raises(RuntimeError,match='mat1 and mat2 shapes'):
            model(torch.zeros(shape,dtype=dtype,device=device))

    @pytest.mark.parametrize('hidden', [(-1,), (2.5,), ('x',), None])
    def test_invalid_hidden_dimensions_rejected(self,hidden):
        with pytest.raises((TypeError,RuntimeError,ValueError)):
            models.MLP(hidden_dims=hidden)
