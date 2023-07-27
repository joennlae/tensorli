import numpy as np


class Moduli:
    def zero_grad(self):
        for p in self.parameters():
            p = np.zeros_like(p)

    def parameters(self):
        return []

class Linearli(Moduli):
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.randn(out_features, in_features)
        # bias activated by default
        self.bias = np.random.randn(out_features)

    def forward(self, x):
        return x @ self.weight.T + self.bias

    def parameters(self):
        return [self.weight, self.bias]
