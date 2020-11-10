from engine_extension import ValueExt
from micrograd.nn import Module
import numpy as np
import random


def _conv(in_matrix, kernel, stride_vertical=1, stride_horizontal=1, padding=0):
    """
    Calculate the convolution of input matrix with kernel. Outputs
    :param in_matrix: a matrix representing input later
    :param kernel: matrix representing kernel values
    :param stride_vertical: vertical kernel stride
    :param stride_horizontal: horizontal kernel stride
    :param padding: padding for input matrix
    :return: a matrix of output values
    """
    k = len(kernel)
    height, width = len(in_matrix), len(in_matrix[0])

    pad_width = padding + width + padding
    pad_height = padding + height + padding

    assert (height * width * k is not 0)  # make sure there are no zero inputs
    assert (len(kernel) % 2 != 0)
    assert (len(in_matrix) == len(in_matrix[0]))

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
        [pad_matrix[r: r + k, c: c + k]
         for c in range(0, pad_width - k + 1, stride_horizontal)]
        for r in range(0, pad_height - k + 1, stride_vertical)]

    # calculate convolution
    return np.array([[np.sum(np.multiply(patch_r, kernel)) for patch_r in patch] for patch in patches])


# Does it matter between random.gauss and random.uniform?
def _build_random_kernel(k, d):
    """
    Build a kernel with random values
    :param k: size
    :param d: depth
    :return:
    """
    return np.array([
        [
            [ValueExt(random.gauss(0, 1)) for _ in range(k)]
            for _ in range(k)]
        for _ in range(d)])


class Conv2D(Module):
    def __init__(self, nin, nout, kernel_size, stride: (int, int), relu=False, padding=0, **kwargs):
        self.nin = nin
        self.nout = nout
        self.kernel_size = kernel_size
        self.stride_vert = stride[0]
        self.stride_horiz = stride[1]
        self.relu = relu
        self.padding = padding
        self.kernels = [_build_random_kernel(kernel_size, nout) for _ in range(nout)]

    def __call__(self, x):  # return a 3 - dim array with output image of convolution for each kernel
        out = np.dstack(
            [_conv(x, kernel, stride_vertical=self.stride_vert,
                   stride_horizontal=self.stride_horiz,
                   padding=self.padding)
             for kernel in self.kernels])
        if self.relu:
            return [
                [
                    [i.relu() for i in row]
                    for row in channel]
                for channel in out]
        return out

    def __parameters__(self):
        parameters = []
        for kernel in self.kernels:
            parameters.append(kernel)
        return parameters

    def __repr__(self):
        return f"Convolutional Layer with  [{len(self.kernels)}] kernels"


# class ConvNet(Module):

#
# Test code for rpow and sigmoid
#

x = ValueExt(2)  # self
y = float(3)  # other

x = y ** x


z = x.sigmoid()
w = ValueExt(2).sigmoid()
print("rpow 2**3 == x == ", x)
print("sigmoid(x) == ", z)
print("sigmoid(2) == ", w)

#
# Test code for _conv
#


# test_in_mat_6x6 = [[3, 2, 3, 4, 5, 15],
#                  [4, 7, -5, 3, 4, -20],
#                 [5, -2, -5, 7, -7, 1],
#                [9, 1, 7, 8, 3, 4],
#               [1, 2, -3, 4, -5, 6],
#              [4, 7, -5, 3, 4, 20]]
# test_in_mat_5x5 = [[3, 2, 3, 4, 5],
#                  [4, 7, 5, 3, 4],
#                 [5, 2, 5, 7, 7],
#                [9, 1, 7, 8, 3],
#               [1, 2, 3, 4, 5]]

# test_in_mat_5x5 = np.array([[Value(i) for i in row] for row in test_in_mat_5x5])

# test_kernel = [[0, 0, 0],
#             [0, 1, 0],
#            [0, 0, 0]]
# test_kernel_1 = _build_random_kernel(3, 1)
# print(test_kernel_1)
# test_out_1 = _conv(test_in_mat_5x5, test_kernel, stride_vertical=1, stride_horizontal=1, padding=1)
# test_out_2 = _conv(test_in_mat_5x5, test_kernel, 1)
# print(test_out_1)
