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
            return Tensor(np.random.randn(input_dim, input_dim) * math.sqrt(2.0 / input_dim))

        self.w_q = [_w_init() for _ in range(heads)]
        self.w_k = [_w_init() for _ in range(heads)]
        self.w_v = [_w_init() for _ in range(heads)]

        self.heads = heads
        self.input_dim = input_dim

    def __call__(self, x: List[Union[Tensor, np.ndarray]]) -> List[Tensor]:
        tokens = Tensor.stack(x)
        q = [Tensor.mat_mul(tokens, self.w_q[h]) for h in range(self.heads)]
        k = [Tensor.mat_mul(tokens, self.w_k[h]) for h in range(self.heads)]
        v = [Tensor.mat_mul(tokens, self.w_v[h]) for h in range(self.heads)]

        
        def scaled_dot_prod(h: int) -> Tensor:
            """
            Compute softmax(qk'/sqrt(d_k))v for head h
            """
            # TODO masking
            # TODO linear bias positional encoding
            query_key_dot = Tensor.one_way_grad_mul(Tensor.mat_mul(q[h], k[h].transpose()), math.sqrt(self.input_dim)**-1)
            
            # TODO softmax
            
            qk_value_dot = Tensor.mat_mul(query_key_dot, v[h])

            return qk_value_dot
            

        return [scaled_dot_prod(h) for h in range(self.heads)]

    def parameters(self):
        return self.w_q + self.w_k + self.w_v


# class TransformerLayer(Module):
#     """
#     Linear layer stacked on Multihead attention
#     """

#     def __init__(self, input_dim: int, output_dim: int, heads: int):
#         self.input_dim = input_dim
#         self.heads = heads
#         self.attn = MultiHeadAttention(input_dim, heads)
#         self.linear = MLP(input_dim * heads, output_dim, hidden_sizes=[16, 16])

#     def __call__(self, x: List[Tensor]) -> List[Tensor]:
#         return [self.linear(token) for token in self.attn(x)]

#     def parameters(self):
#         return self.attn.parameters() + self.linear.parameters()

# c = MultiHeadAttention(3, 1)
# x = [Tensor(np.random.randn(3)) for _ in range(5)]
# y = c(x)
# print(x)

# print('\n')

# print(y)
# y[0].backward()

