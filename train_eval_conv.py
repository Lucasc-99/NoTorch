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
    # MNist train has 60,000 samples
    TRAIN_NUM = 1000  # Number of training batches
    BATCH_SIZE = 60  # Training minibatch size

    TEST_NUM = 1000  # Number of test samples

    #
    # Get data
    #
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])
    train_set = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
    val_set = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
    train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    #
    #

    # Initialize classifier and learning rate hyper-parameter
    classifier = MNistClassifier()
    learning_rate = .01

    #
    # Training loop
    #
    for count, (image_batch, cl_batch) in enumerate(train_dataloader):

        print(f"Running batch:" + str(count))

        classifier.zero_grad()
        probabilities = [softmax(classifier(img)) for img in tqdm(image_batch)]  # Forward pass with softmax

        # Calculate the loss for this batch
        loss = [nll_loss(probabilities[i], cl_batch[i]) for i in range(len(cl_batch))]
        batch_loss = sum(loss)
        batch_loss.backward()

        print("Total loss at batch " + str(count) + " is " + str(batch_loss.data))

        # back-propagate
        for p in classifier.parameters():
            p.data -= learning_rate * p.grad

        if count == TRAIN_NUM - 1:
            break

    print("Training completed, evaluating ")

    correct = 0
    for count, (image, cl) in tqdm(enumerate(val_set)):

        probabilities = softmax(classifier(image))  # Forward pass with softmax
        if probabilities.argmax() == cl:
            correct += 1
        if count == TEST_NUM - 1:
            break

    accuracy = correct / TEST_NUM
    print("Accuracy is: " + str(accuracy))


if __name__ == '__main__':
    main()
