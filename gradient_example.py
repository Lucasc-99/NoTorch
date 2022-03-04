from NoTorch.tensor import Tensor

"""
Gradient example taken directly from micrograd: https://github.com/karpathy/micrograd
"""
a = Tensor(-4.0)
b = Tensor(2.0)

c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f =  e**2
g = f / 2.0
g += 10.0 / f
print(f'{g.data}') # 24.7041
g.backward()
print(f'{a.grad}') # 138.8338
print(f'{b.grad}') # 645.5773








