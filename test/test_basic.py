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
    print(x.grad)

    x_torch = torch.tensor(x_numpy, requires_grad=True)
    y_torch = torch.tensor(y_numpy, requires_grad=True)
    z_torch = x_torch * y_torch + x_torch

    z_torch.backward(torch.ones_like(z_torch))
    print(x_torch.grad)

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

    x_torch = torch.tensor(x_numpy, requires_grad=True)
    y_torch = torch.tensor(y_numpy, requires_grad=True)
    z_torch = x_torch @ y_torch

    z_torch.backward(torch.ones_like(z_torch))

    assert np.allclose(x.data, z_torch.numpy())
    assert np.allclose(x.grad, x_torch.grad.numpy())
