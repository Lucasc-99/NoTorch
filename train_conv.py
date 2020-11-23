from torchvision import datasets, transforms
from engine_extension import Value
from conv import MNistClassifier, softmax, nll_loss


"""
    An example of a single forward 
    and backward pass on the first image in MNist
"""
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])
train_set = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
val_set = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)

image, cl = train_set[0]  # first image only

classifier = MNistClassifier()  # Convolutional NN model for 28x28x1 images
probabilities = softmax(classifier(image))  # Forward pass with softmax
print(probabilities)

# Using Negative Log-Likelihood loss function
loss = nll_loss(probabilities, cl)

classifier.zero_grad()
loss.backward()
params = classifier.parameters()
learning_rate = .001  # This needs to be tuned

# back-propagate
for p in params:
    p.data -= learning_rate * p.grad
