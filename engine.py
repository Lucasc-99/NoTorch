from functools import total_ordering
from typing import Tuple, Callable
import math
import numpy as np
from dataclasses import dataclass


"""
Refactored from: https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py

"""


class Tensor:
    """
    Stores an n-dimensional array and its gradient
    """

    def __init__(self, data: np.ndarray, _children: Tuple) -> None:
        self.data = data
        self._children = _children

        self.grad: np.ndarray = np.zeros_like(data, dtype=data.dtype)
        self._prev: set = set(_children)
        self.backward: Callable = None


    def __add__(self, other):
        other = np.array(other) if not isinstance(other) else other

        out = Tensor(self + other, (self, other))

        def _backward():
            self.grad += out.grad
            

    
    def __rpow__(self, other):
        if not isinstance(other, (int, float, Tensor)):
            return NotImplemented

        other = other if isinstance(other, Tensor) else Tensor(other)
        return other ** self


    def __pow__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data ** other.data, (self, other), f'pow')

        def _backward():
            self.grad += (other.data * self.data ** (other.data - 1)) * out.grad
            if other.data != 0:
                other.grad += (self.data ** other.data) * math.log(abs(other.data)) * out.grad

        out._backward = _backward

        return out


    def sigmoid(self):
        out = Tensor(math.e ** self / (math.e ** self + 1), (self,), f'sigmoid')

        def _backward():
            self.grad += math.e ** self / ((math.e ** self + 1) * (math.e ** self + 1))

        out._backward = _backward

        return out


    def log(self, **kwargs):
        out = Tensor(math.log(self.data), (self,), f'log')

        def _backward():
            self.grad += 1 / self.data * out.grad

        out._backward = _backward

        return out


    def exp(self, *args, **kwargs):
        return math.e ** self


    def __ge__(self, other):
        return self.data >= (other.data if isinstance(other, Tensor) else other)


    def __eq__(self, other):
        return self.data == (other.data if isinstance(other, Tensor) else other)

