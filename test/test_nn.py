import torch
import numpy as np
from tensorli.tensorli import Tensorli
from tensorli.nnli import Linearli


def test_linear():
    in_features = 8
    out_features = 16
    batch_size = 8
    x_numpy = np.random.randn(batch_size, out_features, in_features)
    x = Tensorli(x_numpy)

    linear = Linearli(in_features, out_features, bias=False)
    linear_2 = Linearli(out_features, out_features, bias=True)
    linear_3 = Linearli(out_features, 10)

    out = linear(x)
    out = linear_2(out)
    out = linear_3(out)

    x_torch = torch.tensor(x_numpy, requires_grad=True, dtype=torch.float64)

    linear_torch = torch.nn.Linear(in_features, out_features, bias=False)
    linear_2_torch = torch.nn.Linear(out_features, out_features, bias=True)
    linear_3_torch = torch.nn.Linear(out_features, 10, bias=False)

    linear_torch.weight = torch.nn.Parameter(torch.tensor(linear.weight.data, dtype=torch.float64))

    linear_2_torch.weight = torch.nn.Parameter(
        torch.tensor(linear_2.weight.data, dtype=torch.float64)
    )
    linear_2_torch.bias = torch.nn.Parameter(torch.tensor(linear_2.bias.data, dtype=torch.float64))

    linear_3_torch.weight = torch.nn.Parameter(
        torch.tensor(linear_3.weight.data, dtype=torch.float64)
    )

    out_torch = linear_torch(x_torch)
    out_torch = linear_2_torch(out_torch)
    out_torch = linear_3_torch(out_torch)

    assert np.allclose(out.data, out_torch.detach().numpy())
    out.backward()
    out_torch.backward(torch.ones_like(out_torch))

    assert np.allclose(x.grad.astype(np.float32), x_torch.grad.numpy())
