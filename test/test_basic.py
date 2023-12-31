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


def test_loss_reduction():
    x_numpy = np.random.randn(8, 6, 10)
    x = Tensorli(x_numpy)
    y = x.sum(-1).mean().mean()
    print(y)
    y.backward()

    x_torch = torch.tensor(x_numpy, requires_grad=True)
    y_torch = x_torch.sum(-1).mean()
    print(y_torch)
    y_torch.backward(torch.ones_like(y_torch))

    assert np.allclose(y.data, y_torch.detach().numpy())
    assert np.allclose(x.grad, x_torch.grad.numpy())


def test_mean_bug():
    x_numpy = np.random.randn(1, 6)
    x = Tensorli(x_numpy)
    y = x.mean()
    print(y)
    y.backward()

    x_numpy = np.random.randn(6, 6)
    x = Tensorli(x_numpy)
    y = x.mean().mean()
    print(y)
    y.backward()


def test_cross_entropy():
    x_numpy = np.random.randn(8, 10, 6)
    y_numpy = np.random.randint(0, 10 - 1, size=(8, 6))

    x = Tensorli(x_numpy)
    y = Tensorli(y_numpy)

    x_torch = torch.tensor(x_numpy, requires_grad=True)
    y_torch = torch.tensor(y_numpy, requires_grad=False)

    z_none = x.cross_entropy(y, reduction_type="none")
    z_torch_none = torch.nn.functional.cross_entropy(x_torch, y_torch, reduction="none")
    assert np.allclose(z_none.data, z_torch_none.detach().numpy())

    z_mean = x.cross_entropy(y, reduction_type="mean")
    z_torch_mean = torch.nn.functional.cross_entropy(x_torch, y_torch, reduction="mean")
    assert np.allclose(z_mean.data, z_torch_mean.detach().numpy())

    z_sum = x.cross_entropy(y, reduction_type="sum")
    z_torch_sum = torch.nn.functional.cross_entropy(x_torch, y_torch, reduction="sum")
    assert np.allclose(z_sum.data, z_torch_sum.detach().numpy())

    z_mean.backward()
    z_torch_mean.backward()

    assert np.allclose(x.grad, x_torch.grad.numpy())
