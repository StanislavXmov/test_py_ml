import torch
import matplotlib.pyplot as plt


def func(x):
    return 0.1 * x**2 - torch.sin(x) + 5.0


def model(w, x):
    return w[0] + w[1] * x + w[2] * x**2 + w[3] * x**3


def loss(w, x, y):
    return (model(w, x) - y) ** 2


def dl(w, x, y):
    return 2 * (model(w, x) - y) * torch.tensor([1, x, x**2, x**3], dtype=torch.float32)


coord_x = torch.arange(-5.0, 5.0, 0.1)
coord_y = func(coord_x)

sz = coord_x.size(0)
eta = torch.tensor([0.1, 0.01, 0.001, 0.0001])
w = torch.zeros(4, dtype=torch.float32)
N = 200

for _ in range(N):
    grad = 0
    for i in range(sz):
        grad = grad + dl(w, coord_x[i], coord_y[i])
    w = w - eta * grad / sz

print(w)
q = torch.mean(loss(w, coord_x, coord_y)).item()
print(q)


plt.plot(coord_x, coord_y, label="function")
plt.plot(coord_x, model(w, coord_x), label="model")
plt.legend()
plt.show()
