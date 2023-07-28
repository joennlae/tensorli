import numpy as np
from tensorli.tensorli import Tensorli


class Moduli:
    def zero_grad(self):
        for p in self.parameters():
            p = np.zeros_like(p)

    def parameters(self):
        return []

    def __call__(self, *args, **kwargs) -> Tensorli:
        return self.forward(*args, **kwargs)


class Linearli(Moduli):
    def __init__(self, in_features, out_features, bias=False):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensorli(np.random.randn(out_features, in_features))
        # bias activated by default
        if bias:
            self.bias = Tensorli(np.random.randn(out_features))
        else:
            self.bias = None

    def forward(self, x: Tensorli) -> Tensorli:
        out = x @ self.weight.T
        if self.bias is not None:
            out += self.bias
        return out

    def parameters(self):
        return [self.weight, self.bias]


class Headli(Moduli):
    """one head of self-attention"""

    def __init__(self, embd_dim, seq_len):
        super().__init__()
        self.key = Linearli(embd_dim, embd_dim, bias=False)
        self.query = Linearli(embd_dim, embd_dim, bias=False)
        self.value = Linearli(embd_dim, embd_dim, bias=False)
        self.out_proj = Linearli(embd_dim, embd_dim, bias=False)
        self.where_condition = Tensorli(np.tril(np.ones((seq_len, seq_len))) > 0)
        self.where_neg_inf = Tensorli(np.ones((seq_len, seq_len)) * np.inf * -1)

    def forward(self, x: Tensorli):
        _, _, C = x.shape  # (batch_size, seq_len, embd_dim)
        k = self.key(x)  # (batch_size, seq_len, embd_dim)
        q = self.query(x)  # (batch_size, seq_len, embd_dim)
        # compute attention scores ("affinities")
        # (batch_size, seq_len, embd_dim) @ (batch_size, embd_dim, seq_len)
        # -> (batch_size, seq_len, seq_len)
        wei = q @ k.transpose(-2, -1) * (C**-0.5)
        # this is a decoder as we have a lower triangular with weights
        wei = wei.where(self.where_condition, self.where_neg_inf)
        wei = wei.softmax(-1)  # (batch_size, seq_len, seq_len)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (batch_size, seq_len, embd_dim)
        # (batch_size, seq_len, seq_len) @ (batch_size, seq_len, embd_dim)
        # -> (batch_size, seq_len, embd_dim)
        out = wei @ v
        out = self.out_proj(out)  # (batch_size, seq_len, embd_dim)
        return out
