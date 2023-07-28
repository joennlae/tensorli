from typing import Optional, Union, Tuple, Iterable, Literal
import numpy as np


broadcastedOps = Literal["ADD", "MUL", "DIV", "WHERE"]


class Tensorli:
    def __init__(
        self,
        data: Union[np.ndarray, float, int],
        children: Iterable["Tensorli"] = (),
        op: Optional[str] = "",
        name: Optional[str] = "",
    ) -> None:
        if isinstance(data, (float, int)):
            data = np.array([data])
        self.data = data
        self.grad = np.zeros_like(data)
        self._backward = lambda: None
        self._prev = set(children)

        # debugging
        self._op = op
        self._name = name

    def _broadcasted(
        self, other: "Tensorli", operation: broadcastedOps, condition: Optional["Tensorli"] = None
    ) -> "Tensorli":
        if self.data.shape == other.data.shape:
            return other
        x, y = self, other
        len_x_shape, len_y_shape = len(x.shape), len(y.shape)
        max_shape = max(len_x_shape, len_y_shape)

        # same amount of dimensions
        # common case for matrix multiplication:
        # x = (b, x, 1, y), y = (1, z, y)
        if len_x_shape != max_shape:
            x = x.reshape((1,) * (max_shape - len_x_shape) + x.shape)
        if len_y_shape != max_shape:
            y = y.reshape((1,) * (max_shape - len_y_shape) + y.shape)
        # x = (b, x, 1, y), y = (1, 1, z, y)

        shape_ret = tuple(max(x, y) for x, y in zip(x.shape, y.shape))
        # x = (b, x, 1, y), y = (1, 1, z, y), shape_ret = (b, x, z, y)
        if x.shape != shape_ret:
            x = x.expand(shape_ret)
        if y.shape != shape_ret:
            y = y.expand(shape_ret)
        # TODO: very ugly structure at the moment
        if operation == "ADD":
            return x + y
        if operation == "MUL":
            return x * y
        if operation == "DIV":
            return x / y
        if operation == "WHERE":
            assert condition is not None
            return x.where(condition, y)
        raise NotImplementedError

    def __add__(self, other: Union["Tensorli", "float", "int"]) -> "Tensorli":
        other = other if isinstance(other, Tensorli) else Tensorli(other)
        if self.data.shape != other.data.shape:
            return self._broadcasted(other, "ADD")
        out = Tensorli(self.data + other.data, children=(self, other), op="+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: Union["Tensorli", "float", "int"]) -> "Tensorli":
        other = other if isinstance(other, Tensorli) else Tensorli(other)
        if self.data.shape != other.data.shape:
            return self._broadcasted(other, "MUL")
        # Attention we are using element-wise multiplication!
        # We reshape and expand before the element-wise multiplication
        out = Tensorli(np.multiply(self.data, other.data), children=(self, other), op="*")

        def _backward():
            self.grad += np.multiply(other.data, out.grad)
            other.grad += np.multiply(self.data, out.grad)

        out._backward = _backward
        return out

    def div(self, other: Union["Tensorli", "float", "int"]) -> "Tensorli":
        other = other if isinstance(other, Tensorli) else Tensorli(other)
        if self.data.shape != other.data.shape:
            return self._broadcasted(other, "DIV")

        out = Tensorli(np.divide(self.data, other.data), children=(self, other), op="/")

        def _backward():
            self.grad += np.divide(out.grad, other.data)
            other.grad += np.divide(-self.data * out.grad, other.data**2)

        out._backward = _backward
        return out

    def __pow__(self, other) -> "Tensorli":
        assert isinstance(other, (float, int)), "only supporting int and float powers for now"
        out = Tensorli(self.data**other, children=(self,), op=f"**{other}")

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __neg__(self) -> "Tensorli":
        return self * -1

    def __sub__(self, other) -> "Tensorli":
        return self + (-other)

    def __truediv__(self, other) -> "Tensorli":
        return self.div(other)

    def __repr__(self) -> str:
        return f"Tensorli(data={self.data}, grad={self.grad})"

    def backward(self):
        self.grad = np.ones_like(self.data)  # set to 1.0
        topo = []
        visited = set()

        # build topological order such that we calculate and pass the gradients
        # back in the correct order
        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for child in t._prev:
                    build_topo(child)
                topo.append(t)

        build_topo(self)
        for t in reversed(topo):
            t._backward()

    def print_graph(self, level=0):
        print("  " * level + f"{self._name} {self._op} {self.data.shape}")
        for child in self._prev:
            child.print_graph(level + 1)

    def relu(self) -> "Tensorli":
        out = Tensorli(np.maximum(self.data, 0), children=(self,), op="relu")

        def _backward():
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def expand(self, shape: Tuple[int, ...]):
        if self.data.shape == shape:
            return self
        assert 0 not in shape, f"zeros not allowed in shape {shape}"
        input_shape = self.data.shape
        out = Tensorli(np.broadcast_to(self.data, shape), children=(self,), op="expand")
        if len(input_shape) != len(shape):
            raise NotImplementedError("only supporting same number of dimensions for now")
        axes = []
        for i, (in_dim, out_dim) in enumerate(zip(input_shape, shape)):
            if in_dim != out_dim and (in_dim == 1 or out_dim == 1):
                axes.append(i)

        def _backward():
            self.grad += np.expand_dims(np.sum(out.grad, axis=tuple(axes)), axis=tuple(axes))

        out._backward = _backward
        return out

    def reshape(self, shape: Tuple[int, ...]) -> "Tensorli":
        assert 0 not in shape, f"zeros not allowed in shape {shape}"
        input_shape = self.data.shape
        out = Tensorli(self.data.reshape(shape), children=(self,), op="reshape")

        def _backward():
            self.grad += out.grad.reshape(input_shape)

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdim=False) -> "Tensorli":
        input_shape = self.data.shape
        out = Tensorli(np.sum(self.data, axis=axis, keepdims=keepdim), children=(self,), op="sum")

        def _backward():
            if not keepdim:
                ret = np.expand_dims(out.grad, axis=axis)
            else:
                ret = out.grad
            self.grad += np.broadcast_to(ret, input_shape)

        out._backward = _backward
        return out

    def permute(self, order: list[int]) -> "Tensorli":
        input_order = order
        out = Tensorli(self.data.transpose(order), children=(self,), op="permute")

        def _backward():
            # argsort gives the inverse permutation
            # as this would reshuffle the array back to the initial order
            self.grad += out.grad.transpose(np.argsort(input_order))

        out._backward = _backward
        return out

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def T(self) -> "Tensorli":
        return self.transpose()

    def transpose(self, ax1=-1, ax2=-2) -> "Tensorli":
        order = list(range(len(self.data.shape)))
        order[ax1], order[ax2] = order[ax2], order[ax1]
        return self.permute(order)

    def matmul(self, other: "Tensorli") -> "Tensorli":
        # we do a elementwise multiplication and then sum over the last axis
        # easier for gradient propagation :-)
        # idea is from tinygrad
        n1, n2 = len(self.shape), len(other.shape)
        # (b, x, y) -> (b, x, 1, y) common case
        x = self.reshape((*self.shape[0:-1], *[1] * min(n1 - 1, n2 - 1, 1), self.shape[-1]))
        # (y, z) -> (1, y, z) common case
        other = other.reshape(
            (*other.shape[0:-2], *[1] * min(n1 - 1, n2 - 1, 1), *other.shape[-min(n2, 2) :])
        ).transpose(-1, -min(n2, 2))
        # (1, y, z) -> (transpose) -> (1, z, y)
        # broadcast and then element multiply see _broadcasted
        # (b, x, 1, y), y = (1, 1, z, y) -> (b, x, z, y)
        # sum over y -> (b, x, z)
        return (x * other).sum(-1)

    def __matmul__(self, other: "Tensorli") -> "Tensorli":
        return self.matmul(other)

    def exp(self) -> "Tensorli":
        out = Tensorli(np.exp(self.data), children=(self,), op="exp")

        def _backward():
            self.grad += np.exp(self.data) * out.grad

        out._backward = _backward
        return out

    def softmax(self, axis=-1) -> "Tensorli":
        exp = self.exp()
        out = exp / exp.sum(axis=axis, keepdim=True)
        return out

    def where(self, condition: "Tensorli", other: "Tensorli") -> "Tensorli":
        if self.data.shape != other.data.shape:
            return self._broadcasted(other, "WHERE", condition=condition)
        out = Tensorli(
            np.where(condition.data, self.data, other.data),
            children=(self, other, condition),
            op="where",
        )

        def _backward():
            self.grad += np.where(condition.data, out.grad, np.zeros_like(out.grad))
            other.grad += np.where(condition.data, np.zeros_like(out.grad), out.grad)

        out._backward = _backward
        return out

    def cat(self, other: list["Tensorli"], dim=-1) -> "Tensorli":
        assert len(other) > 0
        assert dim == -1, "only supporting concatenation along last dimension for now"
        assert all(self.shape == o.shape for o in other), "only support same size concat for now"
        out = Tensorli(
            np.concatenate([self.data] + [o.data for o in other], axis=dim),
            children=(self, *other),
            op="cat",
        )

        def _backward():
            self.grad += np.take(out.grad, np.arange(self.shape[dim]), axis=dim)
            for i, o in enumerate(other):
                o.grad += np.take(
                    out.grad, np.arange(self.shape[dim]) + self.shape[dim] * (i + 1), axis=dim
                )

        out._backward = _backward
        return out
