# inspired https://github.com/tinygrad/tinygrad/blob/master/tinygrad/nn/optim.py

import numpy as np
from tensorli.tensorli import Tensorli


class Optimizerli:
    def __init__(self, parameters: list["Tensorli"], lr=0.001):
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad = np.zeros(param.grad.shape)

    def step(self):
        raise NotImplementedError


class Adamli(Optimizerli):  # https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#adam
    def __init__(
        self,
        parameters: list["Tensorli"],
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    ):
        super().__init__(parameters, lr)
        self.parameters = parameters
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros(param.grad.shape) for param in parameters]
        self.v = [np.zeros(param.grad.shape) for param in parameters]
        self.t = 0

    def step(self):
        self.t += 1
        for i, t in enumerate(self.parameters):
            t.grad.clip(-1.0, 1.0, out=t.grad)  # clip gradients very rudiementarily
            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * t.grad
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (t.grad * t.grad)
            m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
            v_hat = self.v[i] / (1 - self.betas[1] ** self.t)
            if self.weight_decay > 0.0:
                t.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * t.data
            else:
                t.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
