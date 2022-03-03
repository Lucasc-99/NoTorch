from NoTorch import nn
import matplotlib.pyplot as plt
import math
import random

"""
Approximate |sin x| with a small neural network

"""


# Define network and sin function
net = MLP(in_features=1, out_features=1, hidden_sizes=[2, 10, 2])
fn = lambda x: math.abs(math.sin(x))


# Define training data
x = [random.randrange(1, 1000) for _ in range(100)]
y = [fn(val) for val in x]


# Define training procedure
learning_rate = 0.001
for sample in x:
    y_pred = net(sample)
    loss = (y_pred - fn(sample)) ** 2


# Find final approximation for 0 - 1000 and plot
apprx = [net(i) for i in range(1000)]
actual = [fn(i) for i in range(1000)]

plt.plot(apprx, list(range(1000)), actual, list(range(1000)))
