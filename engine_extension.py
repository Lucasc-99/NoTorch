from functools import total_ordering

from micrograd.engine import Value
import math


def sigmoid(self):
    out = Value(math.e ** self / (math.e ** self + 1), (self,), f'sigmoid')

    def _backward():
        self.grad += math.e ** self / ((math.e ** self + 1) * (math.e ** self + 1))

    out._backward = _backward

    return out


def ln(self):
    out = Value(math.log(self.data), (self,), f'ln')

    def _backward():
        self.grad += 1 / self.data * out.grad

    out._backward = _backward

    return out


def exp(self):
    return math.e ** self


def __pow__(self, other):
    assert isinstance(other, (int, float)), "only supporting int/float powers for now"
    out = Value(self.data ** other, (self,), f'**{other}')

    def _backward():
        self.grad += (other * self.data ** (other - 1)) * out.grad

    out._backward = _backward

    return out


def __rpow__(self, other):
    out = other ** self.data
    return Value(out)


def __ge__(self: Value, other) -> bool:
    return self.data >= (other.data if isinstance(other, Value) else other)


def __eq__(self: Value, other) -> bool:
    return self.data == (other.data if isinstance(other, Value) else other)


Value.ln = ln
Value.exp = exp

Value.__rpow__ = __rpow__
Value.__pow__ = __pow__
Value.__ge__ = __ge__
Value.__eq__ = __eq__

total_ordering(Value)
