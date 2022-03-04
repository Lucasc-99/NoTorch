from NoTorch.nn import MLP
from NoTorch.tensor import Tensor
import matplotlib.pyplot as plt
import math
import random
import numpy as np
from tqdm import tqdm

"""
Approximate |sin x| with a small neural network

"""


# Define network and sin function
net = MLP(in_features=1, out_features=1, hidden_sizes=[10, 10])
fn = lambda x: 5


# Define training data
x = [random.uniform(0, 1)*10 for _ in range(10000)]
y = [fn(val) for val in x]


# Define training procedure
learning_rate = 0.001
for sample in tqdm(x):
    
    y_pred = net(sample)
    
    loss = (y_pred - fn(sample)) ** 2

    loss.backward()

    for param in net.parameters():
        param.data -= learning_rate * param.grad
    
    net.zero_grad()
    

# Find approximation and plot vs actual

x_plot = np.linspace(0, 10, 10)
apprx = [net(i).data for i in x_plot]
actual = [fn(i) for i in x_plot]

print(apprx)
#plt.plot(x_plot, apprx)

#plt.show()
