import torch
import numpy as np
from tensorli.tensorli import Tensorli


def test_simple():
    x = Tensorli(3)
    y = Tensorli(4)
    z = x * y
    z.backward()
    assert x.grad.data[0] == 4
    assert y.grad.data[0] == 3


def test_simple_comparision():
    x_numpy = np.random.randn(10, 10)
    y_numpy = np.random.randn(10, 10)

    x = Tensorli(x_numpy)
    y = Tensorli(y_numpy)
    z = x * y + x
    z.backward()

    x_torch = torch.tensor(x_numpy, requires_grad=True)
    y_torch = torch.tensor(y_numpy, requires_grad=True)
    z_torch = x_torch * y_torch + x_torch

    z_torch.backward(torch.ones_like(z_torch))

    assert np.allclose(x.grad, x_torch.grad.numpy())


def test_simple_layer():
    x_numpy = np.random.randn(8, 8, 10)
    y_numpy = np.random.randn(6, 10)
    b_numpy = np.random.randn(6)

    x = Tensorli(x_numpy)
    y = Tensorli(y_numpy)
    b = Tensorli(b_numpy)
    z = x @ y.T + b

    x_torch = torch.tensor(x_numpy, requires_grad=True)
    y_torch = torch.tensor(y_numpy, requires_grad=True)
    b_torch = torch.tensor(b_numpy, requires_grad=True)
    z_torch = x_torch @ y_torch.T + b_torch

    z_torch.backward(torch.ones_like(z_torch))
    z.backward()
    assert np.allclose(z.data, z_torch.detach().numpy())
    assert np.allclose(x.grad, x_torch.grad.numpy())


def test_broadcasted():
    x_numpy = np.random.randn(10, 10)
    y_numpy = np.random.randn(10)

    x = Tensorli(x_numpy)
    y = Tensorli(y_numpy)
    z = x + y

    z.backward()

    x_torch = torch.tensor(x_numpy, requires_grad=True)
    y_torch = torch.tensor(y_numpy, requires_grad=True)
    z_torch = x_torch + y_torch

    z_torch.backward(torch.ones_like(z_torch))

    assert np.allclose(x.grad, x_torch.grad.numpy())
    assert np.allclose(y.grad, y_torch.grad.numpy())


def test_expand_reduce():
    x_numpy = np.random.randn(10, 10)
    y_numpy = np.random.randn(10)

    x = Tensorli(x_numpy)
    y = Tensorli(y_numpy)
    z = x.expand((10, 10)) + y

    z.backward()

    x_torch = torch.tensor(x_numpy, requires_grad=True)
    y_torch = torch.tensor(y_numpy, requires_grad=True)
    z_torch = x_torch.expand((10, 10)) + y_torch

    z_torch.backward(torch.ones_like(z_torch))

    assert np.allclose(x.grad, x_torch.grad.numpy())
    assert np.allclose(y.grad, y_torch.grad.numpy())


def test_dot():
    x_numpy = np.random.randn(8, 10, 10)
    y_numpy = np.random.randn(8, 10, 6)

    x = Tensorli(x_numpy)
    y = Tensorli(y_numpy)
    z = x @ y

    z.backward()

    x_torch = torch.tensor(x_numpy, requires_grad=True, dtype=torch.float64)
    y_torch = torch.tensor(y_numpy, requires_grad=True, dtype=torch.float64)
    z_torch = x_torch @ y_torch

    z_torch.backward(torch.ones_like(z_torch))

    assert np.allclose(z.data, z_torch.detach().numpy())
    assert np.allclose(x.grad, x_torch.grad.numpy())


def test_cat():
    xs_numpy = [np.random.randn(8, 6, 10) for _ in range(10)]
    xs = [Tensorli(x) for x in xs_numpy]
    z = xs[0].cat(xs[1:])
    y = z.sum(-1)
    y.backward()

    xs_torch = [torch.tensor(x, requires_grad=True) for x in xs_numpy]
    z_torch = torch.cat(xs_torch, dim=-1)
    y_torch = z_torch.sum(-1)
    y_torch.backward(torch.ones_like(y_torch))

    assert np.allclose(y.data, y_torch.detach().numpy())
    assert np.allclose(xs[0].grad, xs_torch[0].grad.numpy())
    assert np.allclose(xs[3].grad, xs_torch[3].grad.numpy())
