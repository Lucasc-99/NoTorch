<h1>NoTorch</h1>

BROKEN FOR NOW

A 'from scratch' implementation of a deep learning library (no pytorch/tensorflow) built only on NumPy
and heavily inspired by and based on:

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


Visualizing gradients from a forward pass on a single image:

```
$ python test_conv.py
```

Full training and testing:
```
$ python train_eval_conv.py
```

<br>
<h1>Dependencies</h1>

NumPy - https://numpy.org/install/  
micrograd - https://github.com/karpathy/micrograd  
numba - https://numba.pydata.org/numba-doc/latest/user/installing.html  
matplotlib - https://matplotlib.org/  
torchvision (used exclusively for loading MNist data) - https://pypi.org/project/torchvision/  

<br>


<h1>Files</h1>

test_conv.py: single MNist image prediction example, with gradient visualization

train_eval_conv.py: full model training and evaluation

conv.py: contains Convolutional layer class (Conv2D), Convolutional NNetwork class, and an MNist Classifier. 
Also contains negative-log likelyhood loss and softmax functions

engine_extension.py: extends micrograd to add support for log, sigmoid, pow, rpow, e^x, >=, <= operations.
