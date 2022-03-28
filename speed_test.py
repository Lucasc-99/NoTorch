import time
from micrograd.nn import MLP as MicrogradMLP
from NoTorch.nn import MLP as NoTorchMLP


"""
Speed test for Micrograd, NoTorch and PyTorch

The test consists of a forward and backward pass using a MLP from each library
"""

micrograd_model = MicrogradMLP(2, [16, 16, 1])
no_torch_model = NoTorchMLP()


"""
Micrograd
"""
start_micrograd = time.time()
time_micrograd = time.time() - start_micrograd


"""
NoTorch
"""
start_no_torch = time.time()
time_no_torch = time.time() - start_no_torch


"""
PyTorch
"""
start_pytorch = time.time()
time_pytorch = time.time() - start_pytorch


print(
    f"Micrgrad Time: {time_micrograd} \nNoTorch Time: {time_no_torch} \nPyTorch Time: {time_pytorch}"
)
