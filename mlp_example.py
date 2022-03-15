from NoTorch.nn import MLP
from sklearn.datasets import make_moons

"""
Make moons dataset from sklearn

"""

# Define network
net = MLP(in_features=2, out_features=1, hidden_sizes=[16, 16])

# Define training data, normalize y
X, y = make_moons(n_samples=10, noise=0.1)
y = y*2 - 1


# Train
for step in range(100):
    net.zero_grad()
    out = [net(sample) for sample in X]
    losses = [(1.0 + -yi*scorei).relu() for yi, scorei in zip(y, out)]
    # l2_loss = 1e-4 * sum((p*p for p in net.parameters()))

    total_loss = sum(losses)
    accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(y, out)]
    print(sum(accuracy) / len(accuracy))
    total_loss.backward()

    for p in net.parameters():
        print(p.grad)
        p.data -=  p.grad * (1.0 - 0.9 * step /100)

