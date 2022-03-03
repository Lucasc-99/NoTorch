from typing import Tuple, Union
import numpy as np


class Tensor:
    """
    N-dimensional array with differentiable operations

    supports: +, *, /, **, log

    Refactored from https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
    """

    def __init__(
        self, data: Union[np.ndarray, int, float, list], _children: Tuple = ()
    ):
        self.data = data if isinstance(data, np.ndarray) else np.array(data)

        self._children = _children

        self.grad = np.zeros_like(data, dtype=data.dtype)
        self._prev = set(_children)
        self.backward = None

    def __add__(self, other):

        other: Tensor = Tensor._validate_input(other)

        out = Tensor(self.data + other.data, (self, other))

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out.backward = _backward

        return out

    def __mul__(self, other):

        other: Tensor = Tensor._validate_input(other)

        out = Tensor(self.data * other.data, (self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        return out

    def __pow__(self, other):

        other: Tensor = Tensor._validate_input(other)

        out = Tensor(self.data**other.data, (self, other))

        def _backward():
            self.grad += (other.data * self.data ** (other.data - 1)) * out.grad

            # may cause log(0) err
            if np.any(other.data):
                mask = other.data > 0
                other.grad[mask] += (
                    (self.data[mask] ** other.data[mask])
                    * np.log(np.abs(other.data[mask]))
                    * out.grad[mask]
                )

        out._backward = _backward

        return out

    def __rpow__(self, other):
        other: Tensor = Tensor._validate_input(other)

        return other**self

    def sigmoid(self):

        out = Tensor(np.exp(self.data) / (np.exp(self.data) + 1), (self,))

        def _backward():
            self.grad += (
                np.exp(self.data) / ((np.exp + 1) * (np.exp(self.data) + 1)) * out.grad
            )

        out._backward = _backward

        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), f"log")

        def _backward():
            self.grad += (1 / self.data) * out.grad

        out._backward = _backward

        return out

    def relu(self):
        out = Tensor(
            np.zeros_like(self.data) if not np.any(self.data) else self.data, (self)
        )

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

    def __ge__(self, other):
        return self.data >= Tensor._validate_input(other)

    def __eq__(self, other):
        return self.data == Tensor._validate_input(other)

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1



    def backward(self):
        """
        Recursive topological sort of the computation graph,
        Followed by 
        """
        nodes = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                nodes.append(v)
        build_topo(self)


    @staticmethod
    def _validate_input(input):

        if isinstance(input, np.ndarray):
            return Tensor(input, ())

        elif isinstance(input, Tensor):
            return input

        elif isinstance(input, (int, float, list)):
            return Tensor(np.array(input), ())

        else:
            raise NotImplementedError(
                f"Tensor operations for {type(input)} not implemented"
            )
