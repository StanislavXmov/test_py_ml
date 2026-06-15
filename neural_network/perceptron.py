import torch
import matplotlib.pyplot as plt

n = 5
bias = 3

x1 = torch.randn(n)
x2 = x1 + torch.randint(1, 10, [n]) / 10 + bias
c1 = torch.vstack([x1, x2, torch.ones(n)]).mT

x1 = torch.randn(n)
x2 = x1 - torch.randint(1, 10, [n]) / 10 + bias
c2 = torch.vstack([x1, x2, torch.ones(n)]).mT
print(c1)
print(c2)

f = [0 + bias, 1 + bias]
w1 = -0.5
w2 = -w1
w3 = -bias * w2

w = torch.FloatTensor([w1, w2, w3])
for i in range(n):
    x = c1[:][i]
    y = torch.dot(w, x)
    if y > 0:
        print("c1")
    else:
        print("c2")


plt.scatter(c1[:, 0], c1[:, 1], s=10, color="red")
plt.scatter(c2[:, 0], c2[:, 1], s=10, color="blue")
plt.plot(f)
plt.grid(True)
plt.show()
