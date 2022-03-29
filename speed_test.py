import time
from micrograd.nn import MLP as microMLP
import NoTorch
import torch
import numpy as np

"""
Speed test for Micrograd, NoTorch and PyTorch

The test consists of a forward and backward pass using a 4-layer MLP from each library
"""
IN_SIZE = 50
HIDDEN = 200

micrograd_model = microMLP(IN_SIZE, [HIDDEN, HIDDEN, HIDDEN, 1])
no_torch_model = NoTorch.nn.MLP(
    in_features=IN_SIZE, out_features=1, hidden_sizes=[HIDDEN, HIDDEN, HIDDEN]
)
pytorch_model = torch.nn.Sequential(
    torch.nn.Linear(IN_SIZE, HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(HIDDEN, HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(HIDDEN, HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(HIDDEN, 1),
)

x = [np.random.random_sample() for _ in range(50)]


"""
Micrograd
"""
start_micrograd = time.time()
y_micrograd = micrograd_model(x)
y_micrograd.backward()
time_micrograd = time.time() - start_micrograd

"""
NoTorch
"""
start_no_torch = time.time()
y_no_torch = no_torch_model(x)
y_no_torch.backward()
time_no_torch = time.time() - start_no_torch


"""
PyTorch
"""
start_pytorch = time.time()
y_pytorch = pytorch_model(torch.Tensor(x))
y_pytorch.backward()
time_pytorch = time.time() - start_pytorch


print(
    f"Micrograd Time: {time_micrograd} \nNoTorch Time: {time_no_torch} \nPyTorch Time: {time_pytorch}"
)
