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
