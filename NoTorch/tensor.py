from typing import Tuple
import numpy as np


"""
Refactored from: https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py

"""


class Tensor:
    """
    N-dimensional array with differentiable operations

    supports: +, *, /, **, log
    """

    def __init__(self, data: np.ndarray, _children: Tuple):
        self.data = data
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

            # True if other is not zeros
            if np.any(other.data):
                other.grad += (
                    (self.data**other.data) * np.log(np.abs(other.data)) * out.grad
                )

        out._backward = _backward

        return out

    def __rpow__(self, other):
        other: Tensor = Tensor._validate_input(other)

        return other**self

    def sigmoid(self):

        out = Tensor(np.exp(self.data) / (np.exp(self.data) + 1), (self,))

        def _backward():
            self.grad += np.exp(self.data) / ((np.exp + 1) * (np.exp(self.data) + 1)) * out.grad

        out._backward = _backward

        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), f"log")

        def _backward():
            self.grad += (1 / self.data) * out.grad

        out._backward = _backward

        return out

    def __ge__(self, other):
        return self.data >= Tensor._validate_input(other)

    def __eq__(self, other):
        return self.data == Tensor._validate_input(other)

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
