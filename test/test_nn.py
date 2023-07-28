import torch
import numpy as np
from tensorli.tensorli import Tensorli
from tensorli.nnli import Linearli, Headli, MultiHeadAttentionli


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


def test_head():
    embd_dim = 16
    seq_len = 8
    batch_size = 2

    # batch_size, sequence_length, embedding dimensionality
    x_numpy = np.random.randn(batch_size, seq_len, embd_dim)
    x = Tensorli(x_numpy)
    head = Headli(embd_dim, embd_dim, seq_len)

    out = head(x)
    out.backward()

    x_torch = torch.tensor(x_numpy, requires_grad=True, dtype=torch.float64)
    head_torch = torch.nn.MultiheadAttention(embd_dim, 1, bias=False, dropout=0.0, batch_first=True)

    in_proj_numpy = np.concatenate(
        [head.query.weight.data, head.key.weight.data, head.value.weight.data], axis=0
    )
    head_torch.in_proj_weight = torch.nn.Parameter(torch.tensor(in_proj_numpy, dtype=torch.float64))
    head_torch.out_proj.weight = torch.nn.Parameter(
        torch.tensor(head.out_proj.weight.data, dtype=torch.float64)
    )

    attn_mask = torch.tril(torch.ones(seq_len, seq_len)) == 0
    out_torch, _ = head_torch(x_torch, x_torch, x_torch, attn_mask=attn_mask)

    out_torch.backward(torch.ones_like(out_torch))

    assert np.allclose(out.data, out_torch.detach().numpy())
    assert np.allclose(x.grad, x_torch.grad.numpy())


def test_multi_head():
    embd_dim = 64
    seq_len = 32
    batch_size = 8
    n_heads = 4
    np.random.seed(4419)

    # batch_size, sequence_length, embedding dimensionality
    x_numpy = np.random.randn(batch_size, seq_len, embd_dim)
    x = Tensorli(x_numpy)
    multi_head = MultiHeadAttentionli(embd_dim, seq_len, n_heads)

    out = multi_head(x)
    out.backward()

    x_torch = torch.tensor(x_numpy, requires_grad=True, dtype=torch.float64)
    head_torch = torch.nn.MultiheadAttention(
        embd_dim, n_heads, bias=False, dropout=0.0, batch_first=True
    )

    in_proj_numpy = np.concatenate(
        [
            np.concatenate([head.query.weight.data for head in multi_head.heads]),
            np.concatenate([head.key.weight.data for head in multi_head.heads]),
            np.concatenate([head.value.weight.data for head in multi_head.heads]),
        ],
        axis=0,
    )

    head_torch.in_proj_weight = torch.nn.Parameter(torch.tensor(in_proj_numpy, dtype=torch.float64))
    head_torch.out_proj.weight = torch.nn.Parameter(
        torch.tensor(multi_head.out_proj.weight.data, dtype=torch.float64)
    )

    attn_mask = torch.tril(torch.ones(seq_len, seq_len)) == 0
    out_torch, _ = head_torch(x_torch, x_torch, x_torch, need_weights=True, attn_mask=attn_mask)

    out_torch.backward(torch.ones_like(out_torch))

    assert np.allclose(out.data, out_torch.detach().numpy())
    assert np.allclose(x.grad, x_torch.grad.numpy())
