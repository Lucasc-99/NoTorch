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

    def __rpow__(self, other):  # other ** self
        out = ValueExt(1)
        for i in range(self.data):
            out *= other
        return out
