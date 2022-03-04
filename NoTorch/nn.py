from NoTorch.tensor import Tensor
import math
import numpy as np


class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.grad)


class Dense(Module):
    """
    Fully Connected Layer with sigmoid activation
    """

    def __init__(self, in_neurons: int, out_neurons: int, batch_size: int = 1):
        self.in_neurons = in_neurons
        self.out_neurons = out_neurons
        self.weights = [
            Tensor(np.random.randn(in_neurons) * math.sqrt(2.0/in_neurons) )  for _ in range(out_neurons)
        ]
        self.bias = Tensor(np.zeros(shape=[out_neurons]))
        self.batch_size = batch_size

    # TODO: Batching
    def __call__(self, x):
        return Tensor.cat1d([Tensor.sum1d(x * weight) for weight in self.weights]) + self.bias
        

    def parameters(self):
        return [self.bias] + [weight for weight in self.weights]

    def __repr__(self):
        return f'Dense layer with in: {self.in_neurons} out: {self.out_neurons}'

class MLP(Module):
    """
    A basic MLP, with a variable number of hidden layers and sizes
    """

    def __init__(self, in_features: int, out_features: int, hidden_sizes: list):

        if len(hidden_sizes) == 0:
            self.layers = [Dense(in_features, out_features)]

        elif len(hidden_sizes) == 1:
            self.layers = [
                Dense(in_features, hidden_sizes[0]),
                Dense(hidden_sizes[0], out_features),
            ]

        else:
            self.layers = [Dense(in_features, hidden_sizes[0])]
            self.layers += [
                Dense(hidden_sizes[i], hidden_sizes[i + 1])
                for i in range(len(hidden_sizes) - 1)
            ]
            self.layers += [Dense(hidden_sizes[-1], out_features)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params += layer.parameters()
        return params
