<h1>CNN</h1>
A 'from scratch' implementation of a convolutional neural network library (no pytorch/tensorflow) built only on NumPy and Andrej Karpathy's micrograd differentiation engine:

https://github.com/karpathy/micrograd

<h1>Implementation Details</h1>
This project allows users to create convolutional neural networks like the one shown below with an API similar to PyTorch.


![alt text](./conv_net_example.png)



An example classifier for Mnist hand-written digit recognition (smaller than the one above) is provided in conv.py

<br>
<h1>How to Use</h1>
To do a forward pass using the network above, simply call it on an input image like this:


The following example creates a ConvNet with 2 convolutional layers, each with 3 5x5 filter kernels, 
taking 1 input channel, and a ReLU activation function.

```
from conv import ConvNet

conv = ConvNet(in_channels=1,
                            filters=[3, 3],
                            kernel_sizes=[5, 5],
                            activation='relu')

image: np.ndarray # Get an image formatted as a NumPy ndarray

output = conv(image) # this gives us features which can be used for classification

```

<br>

<h1>Files</h1>

test_conv.py: MNist training example

train_eval_conv.py: full model training and evaluation

conv.py: contains Convolutional layer class (Conv2D), Convolutional NNetwork class, and an MNist Classifier. 
Also contains negative-log likelyhood loss and softmax functions

engine_extension.py: extends micrograd to add support for log, sigmoid, pow, rpow, e^x, >=, <= operations.
