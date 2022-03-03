from torchvision import datasets, transforms
from NoTorch.conv import MNistClassifier, softmax, nll_loss
from matplotlib import pyplot as plt

"""
    Broken for now
"""

# Load the data: MNIST digit classification
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])
train_set = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
val_set = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)


# Take the first image only
image, cl = train_set[0]

# Build the model and predict the class of the first image
classifier = MNistClassifier()  # Convolutional NN model for 28x28x1 images
probabilities = softmax(classifier(image))  # Forward pass with softmax
print(f"Predicted digit is {probabilities.argmax()}")

# Compute loss (negative log likelihood)
loss = nll_loss(probabilities, cl)

# Back propagate loss to compute gradients
classifier.zero_grad()
loss.backward()

params = classifier.parameters()

# Update parameters
learning_rate = .001
for p in params:
    p.data -= learning_rate * p.grad


# Visualize Gradients

grad_y = [p.grad for p in params]
grad_x = [i for i in range(len(grad_y))]
s = [2] * len(grad_y)

plt.xlabel('Parameter Index')
plt.ylabel('Parameter Gradient')
plt.title('CNN Model Gradients')

plt.scatter(grad_x, grad_y, s)
plt.show()

