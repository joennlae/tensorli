import numpy as np
from tensorli.tensorli import Tensorli


class Moduli:
    def zero_grad(self):
        for p in self.parameters():
            p = np.zeros_like(p)

    def parameters(self):
        return []

    def __call__(self, *args, **kwargs):
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

    def forward(self, x):
        out = x @ self.weight.T
        if self.bias is not None:
            out += self.bias
        return out

    def parameters(self):
        return [self.weight, self.bias]
