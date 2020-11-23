from functools import total_ordering

from micrograd.engine import Value
import math


def __rpow__(self, other):
    if not isinstance(other, (int, float, Value)):
        return NotImplemented

    other = other if isinstance(other, Value) else Value(other)
    return other ** self


def __pow__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data ** other.data, (self, other), f'pow')

    def _backward():
        self.grad += (other.data * self.data ** (other.data - 1)) * out.grad
        if other.data != 0:
            other.grad += (self.data ** other.data) * math.log(abs(other.data)) * out.grad

    out._backward = _backward

    return out


def sigmoid(self):
    out = Value(math.e ** self / (math.e ** self + 1), (self,), f'sigmoid')

    def _backward():
        self.grad += math.e ** self / ((math.e ** self + 1) * (math.e ** self + 1))

    out._backward = _backward

    return out


def log(self, **kwargs):
    out = Value(math.log(self.data), (self,), f'log')

    def _backward():
        self.grad += 1 / self.data * out.grad

    out._backward = _backward

    return out


def exp(self, *args, **kwargs):
    return math.e ** self


def __ge__(self, other):
    return self.data >= (other.data if isinstance(other, Value) else other)


def __eq__(self, other):
    return self.data == (other.data if isinstance(other, Value) else other)


Value.__rpow__ = __rpow__
Value.__pow__ = __pow__
Value.ln = log
Value.exp = exp
Value.sigmoid = sigmoid
Value.__ge__ = __ge__
Value.__eq__ = __eq__

total_ordering(Value)
