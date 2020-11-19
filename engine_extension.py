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
        out = ValueExt(math.e ** self / (math.e ** self + 1), (self,), f'sigmoid')

        def _backward():
            self.grad += math.e ** self / ((math.e ** self + 1) * (math.e ** self + 1))

        out._backward = _backward

        return out

    def ln(self):
        out = ValueExt(math.log(self.data), (self,), f'ln')

        def _backward():
            self.grad += 1 / self.data * out.grad

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
        out = ValueExt(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def relu(self):
        out = ValueExt(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other ** -1

    def __rtruediv__(self, other):  # other / self
        return other * self ** -1

    def __rpow__(self, other):
        out = other ** self.data
        return ValueExt(out)

    def __ge__(self: Value, other) -> bool:
        return self.data >= (other.data if isinstance(other, Value) else other)

    def __eq__(self: Value, other) -> bool:
        return self.data == (other.data if isinstance(other, Value) else other)