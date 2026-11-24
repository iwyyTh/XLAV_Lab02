"""Independent M3 forward, analytical/numerical gradient and network tests.

Only CPU float32/64 unless CUDA is genuinely available. Reference activations
are used solely as test oracles. ReLU gradcheck points exclude zero.
"""
import math
from unittest.mock import Mock

import pytest
import torch
from torch import nn

import source.activations as acts


@pytest.fixture(params=[torch.float32, torch.float64], ids=['float32','float64'])
def dtype(request):
    return request.param


@pytest.fixture(params=['cpu'] + (['cuda'] if torch.cuda.is_available() else []))
def device(request):
    return torch.device('cuda', torch.cuda.current_device()) if request.param == 'cuda' else torch.device('cpu')


@pytest.fixture(autouse=True)
def preserve_torch_rng():
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        yield


def reference(x, name):
    return torch.relu(x) if name == 'relu' else torch.sigmoid(x)


def close(actual, expected):
    tol = 2e-6 if actual.dtype == torch.float32 else 1e-12
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)


def shaped_values(shape, dtype, device):
    n = math.prod(shape)
    if not shape:
        return torch.tensor(-.7, dtype=dtype, device=device)
    return torch.linspace(-3,3,n,dtype=dtype,device=device).reshape(shape)


class TestReLUForward:
    @pytest.mark.parametrize('shape', [(), (7,), (3,3), (2,2,2,3)], ids=['scalar','vector','matrix','batch'])
    def test_reference_shape_dtype_device_and_nonmutation(self, shape, dtype, device):
        x = shaped_values(shape,dtype,device)
        before = x.clone()
        result = acts.activation_forward(x,'relu')
        assert result.shape == x.shape and result.dtype == dtype and result.device == device
        torch.testing.assert_close(result,torch.relu(x),atol=0,rtol=0)
        assert torch.equal(x,before)

    def test_negative_zero_positive_exact(self, dtype, device):
        x = torch.tensor([-3.,-0.,0.,2.],dtype=dtype,device=device)
        expected = torch.tensor([0.,0.,0.,2.],dtype=dtype,device=device)
        assert torch.equal(acts.activation_forward(x,'relu'),expected)


class TestReLUDerivative:
    @pytest.mark.parametrize('shape', [(),(7,),(3,3),(2,2,2,3)], ids=['scalar','vector','matrix','batch'])
    def test_reference_gradient_and_contract(self, shape, dtype, device):
        x = shaped_values(shape,dtype,device).requires_grad_()
        expected, = torch.autograd.grad(torch.relu(x).sum(),x)
        derivative = acts.activation_derivative(x,'relu')
        assert derivative.shape == x.shape and derivative.dtype == dtype and derivative.device == device
        assert torch.equal(derivative,expected)

    def test_zero_convention_and_sign(self, dtype, device):
        x = torch.tensor([-2.,0.,2.],dtype=dtype,device=device)
        expected = torch.tensor([0.,0.,1.],dtype=dtype,device=device)
        assert torch.equal(acts.activation_derivative(x,'relu'),expected)


class TestSigmoidForward:
    @pytest.mark.parametrize('shape', [(),(7,),(3,3),(2,2,2,3)], ids=['scalar','vector','matrix','batch'])
    def test_reference_shape_dtype_device_and_nonmutation(self, shape, dtype, device):
        x = shaped_values(shape,dtype,device)
        before = x.clone()
        y = acts.activation_forward(x,'sigmoid')
        close(y,torch.sigmoid(x))
        assert y.shape == x.shape and y.dtype == dtype and y.device == device
        assert torch.equal(x,before)

    def test_zero_monotonicity_symmetry_and_extremes(self, dtype, device):
        x = torch.tensor([-1000.,-100.,-8.,-2.,0.,2.,8.,100.,1000.],dtype=dtype,device=device)
        y = acts.activation_forward(x,'sigmoid')
        assert y[4].item() == .5
        assert torch.isfinite(y).all()
        assert ((0 <= y) & (y <= 1)).all()
        assert (y[1:] >= y[:-1]).all()  # Non-strict due to floating-point saturation.
        close(y,torch.sigmoid(x))
        close(acts.activation_forward(-x,'sigmoid'),1-y)

    def test_direct_forward_autograd_stable_at_extremes(self, dtype, device):
        # Checks the graph of activation_forward itself, beyond the custom backward.
        x = torch.tensor([-1000.,-100.,-2.,0.,2.,100.,1000.],dtype=dtype,device=device,requires_grad=True)
        grad, = torch.autograd.grad(acts.activation_forward(x,'sigmoid').sum(),x)
        expected, = torch.autograd.grad(torch.sigmoid(x).sum(),x)
        assert torch.isfinite(grad).all()
        close(grad,expected)
        assert grad[3].item() == .25


class TestSigmoidDerivative:
    def test_reference_bounds_extremes_and_preactivation_contract(self, dtype, device):
        x = torch.tensor([-1000.,-100.,-3.,-1.,0.,1.,3.,100.,1000.],dtype=dtype,device=device,requires_grad=True)
        before = x.detach().clone()
        actual = acts.activation_derivative(x,'sigmoid')
        expected, = torch.autograd.grad(torch.sigmoid(x).sum(),x)
        assert actual.shape == x.shape and actual.dtype == dtype and actual.device == device
        assert torch.isfinite(actual).all()
        assert ((0 <= actual) & (actual <= .25)).all()
        assert actual[4].item() == .25  # derivative(0), not derivative(sigmoid(0)).
        assert 0 < actual[6].item() < .25  # x*(1-x) at x=3 would be -6.
        close(actual,expected)
        assert torch.equal(x.detach(),before)

    def test_independent_central_difference(self, dtype, device, request):
        x = torch.tensor([-5.,-2.,-.4,0.,.7,2.,5.],dtype=dtype,device=device)
        # Evaluate the independent PyTorch oracle in float64, also for float32 input.
        points = x.to(torch.float64)
        eps = 1e-6
        numerical = (torch.sigmoid(points+eps)-torch.sigmoid(points-eps))/(2*eps)
        analytical = acts.activation_derivative(x,'sigmoid').to(torch.float64)
        error = (analytical-numerical).abs().max().item()
        request.node.user_properties.append(('max_abs_gradient_error',error))
        request.node.user_properties.append(('finite_difference_epsilon',eps))
        tolerance = 1e-7 if dtype == torch.float32 else 1e-9
        assert error < tolerance, f'Gradient error {error} exceeds {tolerance}'


class TestCustomAutograd:
    @pytest.mark.parametrize('name', ['relu','sigmoid'])
    @pytest.mark.parametrize('shape', [(),(5,),(2,3),(2,2,2,2)], ids=['scalar','vector','matrix','batch'])
    def test_forward_backward_with_nonunit_upstream(self, name, shape, dtype, device):
        x = shaped_values(shape,dtype,device).requires_grad_()
        before = x.detach().clone()
        upstream = torch.linspace(-2,3,x.numel(),dtype=dtype,device=device).reshape(shape)
        out = acts.CustomActivationFunction.apply(x,name)
        ref = reference(x,name)
        actual, = torch.autograd.grad(out,x,grad_outputs=upstream)
        expected, = torch.autograd.grad(ref,x,grad_outputs=upstream)
        close(out,ref)
        close(actual,expected)
        assert actual.shape == x.shape and actual.dtype == dtype and actual.device == device
        assert torch.equal(x.detach(),before)

    @pytest.mark.parametrize('name', ['relu','sigmoid'])
    def test_gradcheck_float64_away_from_relu_kink(self, name, device):
        x = torch.tensor([[-2.3,-.7,.2],[.8,1.5,3.1]],dtype=torch.float64,device=device,requires_grad=True)
        assert torch.autograd.gradcheck(lambda t: acts.CustomActivationFunction.apply(t,name),
                                        (x,),eps=1e-6,atol=1e-5,rtol=1e-3)

    @pytest.mark.parametrize('name', ['relu','sigmoid'])
    def test_backward_calls_public_derivative_and_uses_its_result(self, name, dtype, device, monkeypatch):
        x = torch.tensor([-2.,0.,3.],dtype=dtype,device=device,requires_grad=True)
        out = acts.CustomActivationFunction.apply(x,name)
        spy = Mock(return_value=torch.full_like(x,.375))
        monkeypatch.setattr(acts,'activation_derivative',spy)
        upstream = torch.tensor([-2.,3.,.5],dtype=dtype,device=device)
        grad, = torch.autograd.grad(out,x,grad_outputs=upstream)
        spy.assert_called_once()
        assert spy.call_args.args[0] is x and spy.call_args.args[1] == name
        assert torch.equal(grad,upstream*.375)

    @pytest.mark.parametrize('name', ['relu','sigmoid'])
    def test_extreme_custom_gradients_finite(self, name, dtype, device):
        x = torch.tensor([-1000.,-100.,0.,100.,1000.],dtype=dtype,device=device,requires_grad=True)
        y = acts.CustomActivationFunction.apply(x,name)
        grad, = torch.autograd.grad(y,x,grad_outputs=torch.tensor([-2.,3.,4.,-.5,2.],dtype=dtype,device=device))
        assert torch.isfinite(y).all() and torch.isfinite(grad).all()
        ref = reference(x,name)
        expected, = torch.autograd.grad(ref,x,grad_outputs=torch.tensor([-2.,3.,4.,-.5,2.],dtype=dtype,device=device))
        close(grad,expected)


class TestCustomModule:
    @pytest.mark.parametrize('name', ['relu','sigmoid'])
    def test_sequential_linear_backprop_and_optimizer_update(self, name, dtype, device):
        def make(activation):
            model = nn.Sequential(nn.Linear(3,4),activation,nn.Linear(4,1)).to(device=device,dtype=dtype)
            # All values fixed; random constructor initializations do not influence assertions.
            with torch.no_grad():
                for layer in (model[0],model[2]):
                    layer.weight.copy_(torch.linspace(.1,.4,layer.weight.numel(),dtype=dtype,device=device).reshape_as(layer.weight))
                    layer.bias.fill_(.1)
            return model
        model = make(acts.CustomActivation(name))
        oracle = make(nn.ReLU() if name=='relu' else nn.Sigmoid())
        x = torch.tensor([[.2,-.1,.7],[.4,.3,-.2]],dtype=dtype,device=device)
        x_ref = x.clone().requires_grad_()
        x.requires_grad_()
        output, expected = model(x),oracle(x_ref)
        close(output,expected)
        assert output.shape == (2,1)
        loss, ref_loss = output.square().mean(),expected.square().mean()
        loss.backward(); ref_loss.backward()
        close(x.grad,x_ref.grad)
        for p,q in zip(model.parameters(),oracle.parameters()):
            assert p.grad is not None and torch.isfinite(p.grad).all()
            assert torch.count_nonzero(p.grad) > 0
            close(p.grad,q.grad)
        before = [p.detach().clone() for p in model.parameters()]
        optimizer = torch.optim.SGD(model.parameters(),lr=.01)
        optimizer.step()
        for p,old in zip(model.parameters(),before):
            assert not torch.equal(p,old)
            close(p.detach(),old-.01*p.grad)


class TestErrorHandling:
    @pytest.mark.parametrize('name', ['tanh','ReLU','',None])
    @pytest.mark.parametrize('entry', ['forward','derivative','module','function'])
    def test_unsupported_name_rejected(self, name, entry):
        x = torch.tensor([-.5,0.,.5])
        with pytest.raises(ValueError,match='Unsupported activation'):
            if entry == 'forward': acts.activation_forward(x,name)
            elif entry == 'derivative': acts.activation_derivative(x,name)
            elif entry == 'module': acts.CustomActivation(name)
            else: acts.CustomActivationFunction.apply(x,name)

    @pytest.mark.parametrize('name', ['relu','sigmoid'])
    @pytest.mark.parametrize('value', [None,[-1.,0.,1.],'invalid'])
    @pytest.mark.parametrize('entry', ['forward','derivative','module'])
    def test_nontensor_input_rejected(self, name, value, entry):
        # Current Tensor-only signature has no promised custom exception/message.
        with pytest.raises(TypeError):
            if entry=='forward': acts.activation_forward(value,name)
            elif entry=='derivative': acts.activation_derivative(value,name)
            else: acts.CustomActivation(name)(value)
