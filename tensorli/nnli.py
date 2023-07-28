import numpy as np
from tensorli.tensorli import Tensorli


class Moduli:
    def zero_grad(self):
        for p in self.parameters():
            p = np.zeros_like(p)

    def parameters(self) -> list[Tensorli]:
        raise NotImplementedError

    def forward(self, *args, **kwargs) -> Tensorli:
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> Tensorli:
        return self.forward(*args, **kwargs)


class Linearli(Moduli):
    def __init__(self, in_features, out_features, bias=False):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensorli(
            np.random.uniform(low=-1.0, high=1.0, size=(out_features, in_features))
        )
        if bias:
            self.bias = Tensorli(np.random.uniform(low=-1.0, high=1.0, size=out_features))
        else:
            self.bias = None

    def forward(self, x: Tensorli) -> Tensorli:
        out = x @ self.weight.T
        if self.bias is not None:
            out += self.bias
        return out

    def parameters(self):
        out = [self.weight]
        if self.bias is not None:
            out.append(self.bias)
        return out


class Headli(Moduli):
    """one head of self-attention"""

    def __init__(self, embd_dim, head_size, seq_len):
        super().__init__()
        self.key = Linearli(embd_dim, head_size, bias=False)
        self.query = Linearli(embd_dim, head_size, bias=False)
        self.value = Linearli(embd_dim, head_size, bias=False)
        self.where_condition = Tensorli(np.tril(np.ones((seq_len, seq_len))) > 0)
        self.where_neg_inf = Tensorli(np.ones((seq_len, seq_len)) * np.inf * -1)
        self.num_heads = embd_dim // head_size

    def forward(self, x: Tensorli) -> Tensorli:
        _, _, C = x.shape  # (batch_size, seq_len, embd_dim)
        k = self.key(x)  # (batch_size, seq_len, head_size)
        q = self.query(x)  # (batch_size, seq_len, head_size)
        # compute attention scores ("affinities")
        # (batch_size, seq_len, head_size) @ (batch_size, head_size, seq_len)
        # -> (batch_size, seq_len, seq_len)
        w = q @ k.transpose(-2, -1) * ((C / self.num_heads) ** -0.5)  # mul is element-wise
        # this is a decoder as we have a lower triangular with weights
        # causal self-attention
        w = w.where(self.where_condition, self.where_neg_inf)
        w = w.softmax(-1)  # (batch_size, seq_len, seq_len)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (batch_size, seq_len, head_size)
        # (batch_size, seq_len, seq_len) @ (batch_size, seq_len, head_size)
        # -> (batch_size, seq_len, head_size)
        out = w @ v
        return out

    def parameters(self) -> list[Tensorli]:
        return self.key.parameters() + self.query.parameters() + self.value.parameters()


class MultiHeadAttentionli(Moduli):
    """Multi-head self-attention"""

    def __init__(self, embd_dim, seq_len, n_heads):
        super().__init__()
        assert embd_dim % n_heads == 0
        self.heads = [Headli(embd_dim, embd_dim // n_heads, seq_len) for _ in range(n_heads)]
        self.out_proj = Linearli(embd_dim, embd_dim, bias=False)

    def forward(self, x: Tensorli) -> Tensorli:
        if len(self.heads) > 1:
            outs = [head(x) for head in self.heads]
            out = outs[0].cat(outs[1:], dim=-1)
        else:
            out = self.heads[0](x)
        out = self.out_proj(out)
        return out

    def parameters(self) -> list[Tensorli]:
        return [p for head in self.heads for p in head.parameters()] + self.out_proj.parameters()
