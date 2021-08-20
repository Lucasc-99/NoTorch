from torchvision import datasets, transforms
from conv import MNistClassifier, softmax, nll_loss
from torch.utils.data import DataLoader
from tqdm import tqdm
from numba import jit

"""
    NOTE: Torch is only used here to load Mnist data

    Full training on MNist and evalutation
    
"""

@jit(forceobj=True)
def main():

    TRAIN_NUM = 1000
    TEST_NUM = 50

    #
    # Get data
    #
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])
    train_set = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
    val_set = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
    train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)
    # test_dataloader = DataLoader(val_set, batch_size=64, shuffle=True)
    #
    #

    # Initialize classifier and learning rate hyper-parameter
    classifier = MNistClassifier()
    learning_rate = .001

    #
    # Training loop
    #
    for count, (image_batch, cl_batch) in enumerate(train_dataloader):

        print(f"\nRunning batch {count}:\n")
        classifier.zero_grad()
        probabilities = [softmax(classifier(img)) for img in tqdm(image_batch)]  # Forward pass with softmax

        exit(0)
        # Using Negative Log-Likelihood loss function

        loss = [nll_loss(probabilities[i], cl_batch[i]) for i in range(len(cl_batch))]

        batch_loss = sum(loss)
        batch_loss.backward()

        print(f"Total loss at batch {count} is {loss}")

        # back-propagate
        for p in classifier.parameters():
            p.data -= learning_rate * p.grad

    correct = 0
    for count, (image, cl) in enumerate(val_set):

        probabilities = softmax(classifier(image))  # Forward pass with softmax
        if probabilities.argmax() == cl:
            correct += 1
        if count == TEST_NUM:
            break

    accuracy = correct / TEST_NUM
    print(f"Accuracy is {accuracy}")

if __name__ == '__main__':
    main()

