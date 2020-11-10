from micrograd.engine import Value


class ValueExt(Value):

    def __init__(self, data, _children=(), _op=''):
        super().__init__(data, _children=(), _op='')
        self.e = 2.71828182845904523536028747135266249775724709369995

    def sigmoid(self):
        out = ValueExt(self.e ** self / (self.e ** self + 1))

        def _backward():
            self.grad += self.e ** self / ((self.e ** self + 1) * (self.e ** self + 1))

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

    def __rpow__(self, other):
        out = other ** self.data
        return ValueExt(out)
