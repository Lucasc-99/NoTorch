<h1>CNN</h1>
A 'from scratch' implementation of a convolutional neural network library (no pytorch/tensorflow) built only on Karpathy's micrograd differentiation engine and NumPy
 

https://github.com/karpathy/micrograd

<h1>Implementaion Details</h1>
This project allows users to create convolutional neural networks with an API similar to PyTorch. An example is provided in which a network is used to achieve high accuracy on Mnist. Also included are softmax and negative log likelihood loss functions.

```
conv = ConvNet(in_channels=1,
                            filters=[3],
                            kernel_sizes=[5],
                            activation='relu')
```

The above example creates a ConvNet with 3 5x5 kernels, taking 1 input channel, including relu as an activation function

<br>
<br>

<h1>Files</h1>

-test_conv.py: MNist training example

-train_eval_conv.py: full model training and evaluation

-conv.py: Contains Convolutional layer class (Conv2D), Convolutional NNetwork class, and an MNist Classifier. 
Also contains negative-log likelyhood loss and softmax functions

-engine_extension.py: An extension of micrograd autodifferentiation to allow for power, log, exp, ==, >=, and sigmoid to be differentiable
