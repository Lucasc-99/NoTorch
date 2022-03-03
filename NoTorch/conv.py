from typing import List, Union
import numpy as np
import random
import sys
from NoTorch.tensor import Tensor




"""

Deprecated

"""






# recursive topological sort requires this
sys.setrecursionlimit(10_000)


def _conv(
    in_matrix: np.ndarray,
    kernel: np.ndarray,
    vertical_stride: int = 1,
    horizontal_stride: int = 1,
    padding: int = None,
):
    """
    Calculate the convolution of input matrix with kernel
    :param in_matrix: a matrix representing input
    :param kernel: matrix representing kernel values
    :param vertical_stride: vertical kernel stride
    :param horizontal_stride: horizontal kernel stride
    :param padding: padding for input matrix
    :return: a matrix of output values
    """
    assert vertical_stride * horizontal_stride > 0  # check for zero stride

    kernel = np.asarray(kernel)
    in_matrix = np.asarray(in_matrix)

    height, width = in_matrix.shape[:2]

    kernel_h = kernel.shape[0]
    kernel_w = kernel.shape[1]

    pad_width = padding + width + padding
    pad_height = padding + height + padding

    # Add padding to the input matrix if necessary

    if padding:
        pad_matrix = [
            [
                in_matrix[i - padding, j - padding]
                if not (
                    i < padding
                    or i >= pad_height - padding
                    or j < padding
                    or j >= pad_width - padding
                )
                else 0
                for j in range(pad_width)
            ]
            for i in range(pad_height)
        ]
    else:
        pad_matrix = in_matrix

    patches = [  # Create a list of image patches
        [
            pad_matrix[r : r + kernel_h, c : c + kernel_w]
            for c in range(0, pad_width - kernel_w + 1, horizontal_stride)
        ]
        for r in range(0, pad_height - kernel_h + 1, vertical_stride)
    ]

    # Calculate each patch and kernel dot product
    # These operations are independent and could be parallelized

    output = [[np.sum(np.multiply(patch, kernel)) for patch in row] for row in patches]

    return np.array(output)


def _build_random_kernel(k, d):
    """
    Build a convolutional kernel with random values,
    :param k: size kxk
    :param d: depth
    :return: kernel
    """
    return np.array(
        [
            [
                # This uses a method based on kaiming initialization
                [Value(random.gauss(0, 1) * np.sqrt(2) / (k * k * d)) for _ in range(d)]
                for _ in range(k)
            ]
            for _ in range(k)
        ]
    )


class Conv2D(Module):
    """
    Convolutional Layer Module
    """

    def __init__(
        self,
        in_channels,
        out_filters,
        kernel_size,
        stride_v=1,
        stride_h=1,
        padding=0,
        activation=None,
    ):
        self.nin = in_channels
        self.nout = out_filters
        self.stride_vert = stride_v
        self.stride_horiz = stride_h
        self.activation = activation
        self.padding = padding
        # Build filters randomly
        self.kernels = [
            _build_random_kernel(kernel_size, in_channels) for _ in range(out_filters)
        ]
        self.activation_fun = None
        self.activation_fun = activation if activation else "None"

    def __call__(
        self, x
    ):  # return a 3 - dim array with output image of convolution for each kernel
        # Pass input matrix through each kernel
        out = np.dstack(
            [
                _conv(x, kernel, self.stride_vert, self.stride_horiz, self.padding)
                for kernel in self.kernels
            ]
        )

        if self.activation_fun == "relu":
            out = [[[i.relu() for i in mat] for mat in channel] for channel in out]

        if self.activation_fun == "sigmoid":
            out = [[[i.sigmoid() for i in row] for row in channel] for channel in out]
        return np.array(out)

    def parameters(self):
        parameters = []
        for kernel in self.kernels:
            parameters.extend(kernel.flat)
        return parameters

    def __repr__(self):
        return f"Convolutional Layer with  [{len(self.kernels)}] kernels"


class ConvNet(Module):
    def __init__(
        self,
        in_channels,
        filters,
        kernel_sizes=None,
        padding_sizes=None,
        activation="None",
    ):
        self.in_channels = in_channels
        self.filters = filters  # number of convolutional layers in the network
        self.size = len(filters)
        self.layers = []
        self.activation = activation
        self.kernel_sizes = (
            kernel_sizes if kernel_sizes else [3 for _ in range(self.size)]
        )
        self.padding_sizes = (
            padding_sizes if padding_sizes else [0 for _ in range(self.size)]
        )

        for i in range(self.size):
            self.layers.append(
                Conv2D(
                    in_channels=self.in_channels,
                    out_filters=self.filters[i],
                    kernel_size=self.kernel_sizes[i],
                    stride_v=1,
                    stride_h=1,
                    padding=self.padding_sizes[i],
                    activation="relu",
                )
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
    """
    A simple classifier for MNist
    using 1 convolutional layer and 1 fully connected layer
    """

    def __init__(self):
        self.classes = 10

        # filters = number of conv layers
        self.conv = ConvNet(
            in_channels=1, filters=[3], kernel_sizes=[5], activation="relu"
        )

        dense_size = 10
        self.dense = MLP(dense_size, [self.classes])

    def __call__(self, img):
        img = img.reshape([28, 28, 1])  # Dimensions specific to MNist

        features = self.conv(img)
        features = features.reshape([-1]).tolist()  # do we need to do .tolist here?

        pred = self.dense(features)

        out = np.array(pred)
        return out

    def parameters(self):
        return self.conv.parameters() + self.dense.parameters()


def softmax(in_vector: Union[List, np.ndarray]) -> np.ndarray:
    """
    Softmax normalization function
    :param in_vector: a 1-dim vector
    :return: a 1-dim vector with normalized values
    """
    x = np.asarray(in_vector)
    x -= x.max()

    out = np.exp(x)
    out /= out.sum(-1)
    return out


def nll_loss(probabilities: Union[List, np.ndarray], cl: int) -> Value:
    """
    Negative Log Likelihood Loss function
    :param probabilities: the probabilities of each class
    :param cl: the index of the correct class
    :return: a loss value
    """
    return (
        (-1 * probabilities[cl].log()) if probabilities[cl] != 0 else probabilities**0
    )
