<h1>CNN</h1>
A 'from scratch' implementation of a simple convolutional neural network (no pytorch/tensorflow) built only on Karpathy's micrograd differentiation engine

https://github.com/karpathy/micrograd

<h1>Files</h1>

-test_conv.py: MNist training example

-train_eval_conv.py: full model training and evaluation

-conv.py: Contains Convolutional layer class (Conv2D), Convolutional NNetwork class, and an MNist Classifier. 
Also contains negative-log likelyhood loss and softmax functions

-engine_extension.py: An extension of micrograd autodifferentiation to allow for power, log, exp, ==, >=, and sigmoid to be differentiable
