"""
Attention for Transformers
"""

from typing import List, Union
from NoTorch.nn import Module
from NoTorch.tensor import Tensor
import numpy as np
import math


class MultiHeadAttention(Module):
    """
    Multi-Head Scaled Dot Product Attention
    """

    def __init__(self, input_dim: int, heads: int):

        def _w_init() -> Tensor:
            return Tensor(np.array(
                [
                    np.longdouble(np.random.randn(input_dim))
                    * math.sqrt(2.0 / input_dim)
                    for _ in range(input_dim)
                ]
            ))
        
        self.w_query = [_w_init() for _ in heads]
        self.w_key = [_w_init() for _ in heads]
        self.w_value = [_w_init() for _ in heads]
    
    def __call__(self, x: List[Union[Tensor, np.ndarray]]) -> List[Tensor]:
        # TODO
        pass

    def parameters(self):
        return [self.w_query, self.w_key, self.w_value]
