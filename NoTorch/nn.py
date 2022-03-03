from ast import Mod
from typing import Callable
from tensor import Tensor

import numpy as np


"""
Refactored from: https://github.com/karpathy/micrograd/blob/master/micrograd/nn.py

"""


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
    
    def parameters(self):
        return [self.weight, self.bias]
