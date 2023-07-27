import numpy as np
from typing import Optional, Union

class Tensorli():
    def __init__(self, data: Union[np.ndarray, float, int], children: Optional[tuple[np.ndarray, np.ndarray]]=(), op: Optional[str] = '') -> None:
        self.grad : Optional[Tensorli] = None
        if isinstance(data, (float, int)):
            data = np.array([data])
        self.data = data
        self._backward = lambda: None
        self._parents = set(children)

        # debugging
        self._op = op

    def __add__(self, other):
        other = other if isinstance(other, Tensorli) else Tensorli(other)
        out = Tensorli(self.data + other.data, children=(self, other), op = "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensorli) else Tensorli(other)
        out = Tensorli(self.data * other.data, children=(self, other), op = "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (float, int)), "only supporting int and float powers for now"
        out = Tensorli(self.data ** other, children=(self,), op = f"**{other}")

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other ** -1

    def __repr__(self):
        return f"Tensorli(data={self.data}, grad={self.grad})"
