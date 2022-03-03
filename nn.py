from typing import Callable
from tensor import Tensor

import numpy as np

class Dense:
    """
    Fully Connected Layer
    """

    def __init__(self, in_neurons: int, out_neurons: int, activation: Callable):
        self.weight = Tensor(np.zeros(in_neurons), ())
        self.bias = Tensor(np.zeros(out_neurons), ())