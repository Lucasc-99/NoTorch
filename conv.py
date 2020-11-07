from micrograd.engine import Value
from micrograd.nn import Module, Neuron, Layer, MLP
import numpy as np
import random

KERNEL_SIZE = 3
STRIDE = 1
PADDING = 0


def _conv(in_matrix, kernel, stride):
    """
    VERY inefficient code for calculating output of convolutional layer
    :param in_matrix: a matrix representing input later
    :param kernel: matrix representing kernel values
    :param stride: step size for kernel
    :return: a matrix of output values, linearized with relu
    """
    assert (len(in_matrix) * len(kernel) * stride is not 0)  # make sure there are no zero inputs
    assert (len(kernel) == len(kernel[0]))  # kernel must be square
    assert (len(in_matrix) == len(in_matrix[0]))
    bound = len(in_matrix) - len(kernel) + 1
    out_matrix = np.empty(shape=(bound, bound))
    out_matrix.fill(0)
    i = 0
    j = 0
    for i in range(0, len(in_matrix) - len(kernel) + 1):
        for j in range(0, len(in_matrix[0]) - len(kernel) + 1):
            for x in range(0, len(kernel)):
                for y in range(0, len(kernel)):
                    out_matrix[i][j] += kernel[x][y] * in_matrix[x + i][y + j]
            out_matrix[i][j] *= (out_matrix[i][j] > 0)  # relu

    return out_matrix


#
# Test code for _conv
#


test_in_mat_6x6 = [[3, 2, 3, 4, 5, 15],
                   [4, 7, 5, 3, 4, 20],
                   [5, 2, 5, 7, 7, 1],
                   [9, 1, 7, 8, 3, 4],
                   [1, 2, 3, 4, 5, 6],
                   [4, 7, 5, 3, 4, 20]]
test_in_mat_5x5 = [[3, 2, 3, 4, 5],
                   [4, 7, 5, 3, 4],
                   [5, 2, 5, 7, 7],
                   [9, 1, 7, 8, 3],
                   [1, 2, 3, 4, 5]]

test_kernel = [[1, 0, 0],
               [0, 1, 0],
               [0, 0, 1]]

test_out_1 = _conv(test_in_mat_6x6, test_kernel, 1)
test_out_2 = _conv(test_in_mat_5x5, test_kernel, 1)
print(test_out_1)
print(test_out_2)


class Conv2D(Module):
    """
        Each neuron in this layer maps to size_kernel ^ 2 other neurons
        There should be overlap on these neurons?
        How do I connect this to an input layer????
    """

    def __init__(self, nin, nout, kernel, **kwargs):
        self.kernel = kernel
        self.neurons = [Neuron(KERNEL_SIZE ** 2, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

# class ConvNetwork(MLP):
#   def __init__(self, nin, conv_layers, nout):
