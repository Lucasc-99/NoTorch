import time
from micrograd.nn import MLP as microMLP
from notorch.nn import MLP as no_torch_MLP
import torch
import numpy as np
import json
import os


IN_SIZE = list(range(20, 200, 10))
HIDDEN = [insize * 2 for insize in IN_SIZE]
RESULTS_PATH = "results/"
EXPERIMENT_TAG = "speed_test.jsonl"


print(
    f"""Speed Test:
For each run, each library executes a forward and backward pass on the same random data using an 8-layer MLP
"""
)

results = []
for insize, hidden in zip(IN_SIZE, HIDDEN):

    micrograd_model = microMLP(insize, [hidden] * 7 + [1])
    no_torch_model = no_torch_MLP(
        in_features=insize,
        out_features=1,
        hidden_sizes=[hidden] * 7,
    )
    layers = (
        [torch.nn.Linear(insize, hidden), torch.nn.ReLU()]
        + [torch.nn.Linear(hidden, hidden), torch.nn.ReLU()] * 6
        + [torch.nn.Linear(hidden, 1)]
    )
    pytorch_model = torch.nn.Sequential(*layers)

    x = [np.float32(np.random.random_sample()) for _ in range(insize)]

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
        f"Micrograd Forward/Backward Time: {time_micrograd} seconds \nNoTorch Forward/Backward Time: {time_no_torch} seconds \nPyTorch Forward/Backward Time: {time_pytorch} seconds"
    )
    result = {
        "input_size": insize,
        "hidden_size": hidden,
        "micrograd_time": time_micrograd,
        "no_torch_time": time_no_torch,
        "pytorch_time": time_pytorch,
    }
    results.append(result)

    # Print a separator for readability in the console output
    print("-" * 80)


#
# Save and display results
#

os.makedirs(RESULTS_PATH, exist_ok=True)

with open(os.path.join(RESULTS_PATH, EXPERIMENT_TAG), "w") as f:
    for result in results:
        f.write(json.dumps(result) + "\n")

print(f"Results saved to {os.path.join(RESULTS_PATH, EXPERIMENT_TAG)}")

micrograd_avg = sum(r["micrograd_time"] for r in results) / len(results)
no_torch_avg = sum(r["no_torch_time"] for r in results) / len(results)
pytorch_avg = sum(r["pytorch_time"] for r in results) / len(results)

print(f"\nAverage times across all runs:")
print(f"Micrograd: {micrograd_avg:.6f} seconds")
print(f"NoTorch: {no_torch_avg:.6f} seconds")
print(f"PyTorch: {pytorch_avg:.6f} seconds")
