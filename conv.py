import math

from micrograd.nn import Module, MLP
from typing import List, Union
import numpy as np
import random
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from engine_extension import Value

# from numba import  jit, cuda


def _conv(in_matrix, kernel, vertical_stride=1, horizontal_stride=1, padding=0):
    """
    Calculate the convolution of input matrix with kernel. Outputs
    :param in_matrix: a matrix representing input later
    :param kernel: matrix representing kernel values
    :param vertical_stride: vertical kernel stride
    :param horizontal_stride: horizontal kernel stride
    :param padding: padding for input matrix
    :return: a matrix of output values
    """
    kernel = np.asarray(kernel)
    in_matrix = np.asarray(in_matrix)

    height, width = in_matrix.shape[:2]

    kernel_h = kernel.shape[0]
    kernel_w = kernel.shape[1]

    pad_width = padding + width + padding
    pad_height = padding + height + padding

    if padding:
        pad_matrix = [
            [in_matrix[i - padding, j - padding]
             if not (i < padding or i >= pad_height - padding or j < padding or j >= pad_width - padding)
             else 0 for j in range(pad_width)] for i in range(pad_height)]
    else:
        pad_matrix = in_matrix

    return [
        [
            np.sum(np.multiply(pad_matrix[row: row + kernel_h, col: col + kernel_w], kernel))
            for col in range(0, width - kernel_w + 1, horizontal_stride)
        ]
        for row in range(0, height - kernel_h + 1, vertical_stride)
    ]


def _build_random_kernels(k, d):
    """
    Build a kernel with random values
    :param k: size kxk
    :param d: depth
    :return: kernel
    """
    return np.array([
        [
            [Value(random.gauss(0, 1)) for _ in range(d)]
            for _ in range(k)]
        for _ in range(k)])


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
        self.activation_fun = activation if activation else 'None'

    def __call__(self, x):  # return a 3 - dim array with output image of convolution for each kernel
        out = np.dstack(
            [_conv(x, kernel, self.stride_vert, self.stride_horiz, self.padding)
             for kernel in self.kernels])

        if self.activation_fun == 'relu':
            out = [
                [
                    [i.relu() for i in mat]
                    for mat in channel]
                for channel in out]

        if self.activation_fun == 'sigmoid':
            out = [
                [
                    [i.sigmoid() for i in row]
                    for row in channel]
                for channel in out]
        return np.array(out)

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
        self.layers = []
        self.activation = activation
        self.kernel_sizes = kernel_sizes if kernel_sizes else [3 for _ in range(self.size)]
        self.padding_sizes = padding_sizes if padding_sizes else [1 for _ in range(self.size)]

        for i in range(self.size):
            self.layers.append(
                Conv2D(self.in_channels,
                       self.filters[i],
                       self.kernel_sizes[i],
                       self.padding_sizes[i],
                       activation='relu')
            )

    def __call__(self, x):
        for conv_2d in self.layers:
            x = conv_2d(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"Convolutional Network with {self.in_channels} inputs and {self.size} filters"


class MNistClassifier(Module):
    def __init__(self, classes):
        self.classes = classes

        self.conv = ConvNet(in_channels=1,
                            filters=[4, 4, 4, 4, 4],
                            kernel_sizes=[5, 5, 3, 3, 3],
                            activation='relu')

        dense_size = 784  # 28 * 28?
        self.dense = MLP(dense_size, [classes])

    def __call__(self, img):
        img = img.reshape([28, 28, 1])  # How do these dimensions change?
        features = self.conv(img)
        features = features.reshape([-1]).tolist()
        return self.dense(features)

    def parameters(self):
        return self.conv.parameters() + self.dense.parameters()


def softmax(in_vector: Union[List, np.ndarray]) -> np.ndarray:
    in_vector = np.asarray(in_vector)
    in_vector -= in_vector.max()
    in_vector = np.exp(in_vector)
    return in_vector / in_vector.sum(-1)


if __name__ == '__main__':
    """ Using this: 
    
    https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627  
    
    tutorial"""

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])

    trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
    valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)

    im_test, cl_test = trainset[0]
    classifier = MNistClassifier(10)
    out = classifier(im_test)
    loss = -(softmax(out).max().ln())
    loss.backward()
    print(loss.grad)
