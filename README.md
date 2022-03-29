<h1>NoTorch</h1>

A 'from scratch' implementation of a deep learning library (no pytorch/tensorflow) built only on NumPy.

This is a learning project heavily inspired by and based on:

https://github.com/karpathy/micrograd

Just like micrograd, this project contains a gradient engine and a neural network library

Additional features include:

- A matrix valued autograd engine, allowing for differentiable matrix operations:
    - element-wise: +, -, *, /, ^, log, exp, relu, sigmoid
    - matrix multiplication (used for efficient forward passes)
    - summation and concatenation across a single dimension

- Extremely fast performance, speeds similar to PyTorch and orders of magnitude faster than micrograd

IN PROGRESS:

- Attention layers for Transformers
- GPU support via CuPy


<h1>Implementation Details</h1>

Coming soon



<br>
<h1>How to Use</h1>

To run mlp_example.py: 
```
$ pip install poetry
$ git clone https://github.com/Lucasc-99/NoTorch
$ cd NoTorch
$ poetry install 
$ poetry run python mlp_example.py
```