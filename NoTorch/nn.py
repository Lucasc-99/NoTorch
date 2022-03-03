from NoTorch.tensor import Tensor

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
        self.weight = Tensor(np.zeros(shape=[in_neurons, out_neurons]), ())
        self.bias = Tensor(np.zeros(shape=[out_neurons]), ())
        self.batch_size = batch_size

    def __call__(self, x: Tensor):
        # TODO: Batching
        print(x)
        print(self.weight)
        print(self.bias)

        out =  (x * self.weight + self.bias).sigmoid()
        print("Calculation succesful")
        print(out)
        return out
        

    def parameters(self):
        return [self.weight, self.bias]


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