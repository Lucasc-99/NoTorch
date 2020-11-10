from micrograd.engine import Value
import math


class ValueExt(Value):
    """
        Extension class for micrograd.engine.Value
        mul, add, pow, relu methods copied directly from Value class
    """
    def __init__(self, data, _children=(), _op=''):
        super().__init__(data, _children=(), _op='')

    def sigmoid(self):
        out = ValueExt(math.e ** self / (math.e ** self + 1))

        def _backward():
            self.grad += math.e ** self / ((math.e ** self + 1) * (math.e ** self + 1))

        out._backward = _backward

        return out

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = ValueExt(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = ValueExt(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = ValueExt(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = ValueExt(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def __rpow__(self, other):
        out = other ** self.data
        return ValueExt(out)
