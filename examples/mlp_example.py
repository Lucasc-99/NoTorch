from black import out
from ..NoTorch.nn import MLP
import math


net = MLP(in_features=2, out_features=2, hidden_sizes=[3, 3])

fn = lambda x: math.sin(x)







