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

    def __init__(self, in_neurons: int, out_neurons: int, batch_size: int = 1):
        self.weight = Tensor(np.zeros(shape=[in_neurons, out_neurons]), ())
        self.bias = Tensor(np.zeros(shape=[out_neurons]), ())
        self.batch_size = batch_size


    def __call__(self, x: Tensor):
        # TODO: Batching
        return (x*self.weight + self.bias).sigmoid()

    def parameters(self):
        return [self.weight, self.bias]
