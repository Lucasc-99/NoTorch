from typing import Callable
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
    Fully Connected Layer
    """

    def __init__(self, in_neurons: int, out_neurons: int, activation: Callable):
        self.weight = Tensor(np.zeros(in_neurons), ())
        self.bias = Tensor(np.zeros(out_neurons), ())
        self.activation = activation
    
    def __call__(self, x: Tensor):
        pass


    def parameters(self):
        return [self.weight, self.bias]

class ReLU(Module):
    """
    Rectified Linear Unit (ReLU) activation function
    """

    def __call__(self, x: Tensor):
        pass