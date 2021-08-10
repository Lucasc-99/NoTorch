from torchvision import datasets, transforms
from conv import MNistClassifier, softmax, nll_loss

"""
    Full training on MNist and evalutation
    
"""

TRAIN_NUM = 1000
TEST_NUM = 50
if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])
    train_set = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
    val_set = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)

    classifier = MNistClassifier()  # Convolutional NN model for 28x28x1 images
    learning_rate = .001  # This needs to be tuned

    for count, (image, cl) in enumerate(train_set):
        probabilities = softmax(classifier(image))  # Forward pass with softmax

        # Using Negative Log-Likelihood loss function
        loss = nll_loss(probabilities, cl)
        print(f"Loss at {count} is {loss}")
        classifier.zero_grad()
        loss.backward()
        # back-propagate
        for p in classifier.parameters():
            p.data -= learning_rate * p.grad
        if count == TRAIN_NUM:
            break

    correct = 0
    for count, (image, cl) in enumerate(val_set):
        probabilities = softmax(classifier(image))  # Forward pass with softmax
        if probabilities.argmax() == cl:
            correct += 1
        if count == TEST_NUM:
            break

    accuracy = correct / TEST_NUM
    print(f"Accuracy is {accuracy}")

