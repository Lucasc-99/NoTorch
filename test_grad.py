from NoTorch.tensor import Tensor



a = Tensor([-4.0, -4.0, -4.0, -4.0])
b = Tensor([2.0, 2.0, 2.0, 2.0])



c = a + b
d = a * b + b**Tensor([3] * 4)
c += c + Tensor([1] * 4)
c += Tensor([1] * 4) + c + (-a)
d += d * Tensor([2] * 4) + (b + a).relu()

d += Tensor([3] * 4) * d + (b - a).relu()

e = c - d
f = e**Tensor([2] * 4)
g = f / Tensor([2] * 4)
g += Tensor([10.0] * 4) / f

print(f"{g.data}")  # 24.7041
g.backward()

print(f"{a.grad}")  # 138.8338
print(f"{b.grad}")  # 645.5773
