from NoTorch.tensor import Tensor

import numpy as np


class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.grad)


class Dense(Module):
    """
    Fully Connected Layer with sigmoid activation
    """

    def __init__(self, in_neurons: int, out_neurons: int):
        self.weight = Tensor(np.zeros(shape=[1, in_neurons, out_neurons]), ())
        self.bias = Tensor(np.zeros(shape=[1, out_neurons]), ())

    def __call__(self, x: Tensor):
        
        return x.sigmoid()

    def parameters(self):
        return [self.weight, self.bias]
