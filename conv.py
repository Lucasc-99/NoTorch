from engine_extension import ValueExt
from micrograd.nn import Module, MLP
import numpy as np
import random


def _conv(in_matrix, kernel, vertical_stride=1, horizontal_stride=1, padding=0):
    """
    Calculate the convolution of input matrix with kernel. Outputs
    :param in_matrix: a matrix representing input later
    :param kernel: matrix representing kernel values
    :param stride_vertical: vertical kernel stride
    :param stride_horizontal: horizontal kernel stride
    :param padding: padding for input matrix
    :return: a matrix of output values
    """
    k = len(kernel[0])
    height, width = len(in_matrix), len(in_matrix[0])

    in_matrix = np.asarray(in_matrix)

    pad_width = padding + width + padding
    pad_height = padding + height + padding

    assert (height * width * k is not 0)  # make sure there are no zero inputs
    assert (len(kernel) % 2 != 0)
    assert (len(in_matrix) == len(in_matrix[0]))

    pad_matrix = None
    if padding:
        pad_matrix = np.empty(shape=(pad_height, pad_width))
        for i in range(pad_height):
            for j in range(pad_width):
                if not (i < padding or i >= pad_height - padding or j < padding or j >= pad_width - padding):
                    pad_matrix[i][j] = in_matrix[i - padding][j - padding]
                else:
                    pad_matrix[i][j] = 0
    else:
        pad_matrix = in_matrix

    # find patches
    patches = [
        [
            pad_matrix[start_row: start_row + k, start_col: start_col + k]
            for start_col in range(0, width - k + 1, horizontal_stride)
        ]
        for start_row in range(0, height - k + 1, vertical_stride)
    ]

    # calculate convolution
    return np.array([[np.sum(np.multiply(patch_r, kernel)) for patch_r in patch] for patch in patches])


# Does it matter between random.gauss and random.uniform?
def _build_random_kernels(k, d):
    """
    Build a kernel with random values
    :param k: size
    :param d: depth
    :return: kernel
    """
    return np.array([
        [
            [ValueExt(random.gauss(0, 1)) for _ in range(k)]
            for _ in range(k)]
        for _ in range(d)])


class Conv2D(Module):
    def __init__(self, in_channels, out_filters, kernel_size, stride_v=1, stride_h=1, padding=0, activation=None):
        self.nin = in_channels
        self.nout = out_filters
        self.stride_vert = stride_v
        self.stride_horiz = stride_h
        self.activation = activation
        self.padding = padding
        self.kernels = [_build_random_kernels(kernel_size, in_channels) for _ in range(out_filters)]
        self.activation_fun = None
        if activation:
            if activation == 'relu':
                self.activation_fun = ValueExt.relu
            elif activation == 'sigmoid':
                self.activation_fun = ValueExt.sigmoid

    def __call__(self, x):  # return a 3 - dim array with output image of convolution for each kernel
        out = np.dstack(
            [_conv(x, kernel, self.stride_vert, self.stride_horiz, self.padding)
             for kernel in self.kernels])
        if self.activation_fun:
            out = [
                [
                    [i.relu() for i in row]  # Python not functional enough for this?
                    for row in channel]
                for channel in out]

        return out

    def parameters(self):
        parameters = []
        for kernel in self.kernels:
            parameters.append(kernel)
        return parameters

    def __repr__(self):
        return f"Convolutional Layer with  [{len(self.kernels)}] kernels"


class ConvNet(Module):
    def __init__(self, in_channels, filters, kernel_sizes=None, padding_sizes=None, activation='None'):
        self.in_channels = in_channels
        self.filters = filters  # number of layers in the network
        self.size = len(filters)
        self.layer_list = []
        self.activation = activation
        self.kernel_sizes = kernel_sizes if kernel_sizes else [3 for _ in range(self.size)]
        self.padding_sizes = padding_sizes if padding_sizes else [1 for _ in range(self.size)]

        for i in range(self.size):
            self.layer_list.append(
                Conv2D(self.in_channels,
                       self.filters[i],
                       self.kernel_sizes[i],
                       self.padding_sizes[i],
                       activation='relu')
            )

    def __call__(self, x):
        for convolutional_layer in self.layer_list:
            x = convolutional_layer(x)
        return x

    def parameters(self):
        return [layer.parameters() for layer in self.layer_list]

    def __repr__(self):
        return f"Convolutional Network with {self.in_channels} inputs and {self.size} filters"


c = ConvNet(1, [3], activation='relu')
test_in_mat_6x6 = [[3, 2, 3, 4, 5, 15],
                  [4, 7, -5, 3, 4, -20],
                 [5, -2, -5, 7, -7, 1],
                [9, 1, 7, 8, 3, 4],
               [1, 2, -3, 4, -5, 6],
              [4, 7, -5, 3, 4, 20]]

a = c(test_in_mat_6x6)

print(len(a))
#
# Test code for rpow and sigmoid
#

"""
x = ValueExt(2)  # self

y = float(3)  # other
x = y ** x
z = x.sigmoid()
w = ValueExt(2).sigmoid()

a = ValueExt(-20).relu()
print(type(a))
print("rpow 2**3 == x == ", x)
print("sigmoid(x) == ", z)
print("sigmoid(2) == ", w)
"""

#
# Test code for _conv
#
'''

test_in_mat_6x6 = [[3, 2, 3, 4, 5, 15],
                  [4, 7, -5, 3, 4, -20],
                 [5, -2, -5, 7, -7, 1],
                [9, 1, 7, 8, 3, 4],
               [1, 2, -3, 4, -5, 6],
              [4, 7, -5, 3, 4, 20]]

test_in_mat_5x5 = [[3, 2, 3, 4, 5],
                  [4, 7, 5, 3, 4],
                 [5, 2, 5, 7, 7],
                [9, 1, 7, 8, 3],
               [1, 2, 3, 4, 5]]


test_kernel = [[0, 0, 0],
             [0, 1, 0],
            [0, 0, 0]]

test_kernel_1 = _build_random_kernel(3, 1)
print(test_kernel_1)
test_out_1 = _conv(test_in_mat_5x5, test_kernel, 1, 1, padding=1)
test_out_2 = _conv(test_in_mat_5x5, test_kernel, 1)
print(test_out_1)
'''