import torch
import torch.optim as optim


def func(x):
    return 0.2 * (x - 2) ** 2 - 0.3 * torch.cos(4 * x)


x0 = 0.0
lr = 0.01
n = 200

x = torch.tensor([x0], requires_grad=True)

optimizer = optim.Adam(params=[x], lr=0.01)

for _ in range(n):
    y = func(x)
    y.backward()
    # x.data = x.data - lr * x.grad
    # x.grad.zero_()
    optimizer.step()
    optimizer.zero_grad()

print(x.item())
