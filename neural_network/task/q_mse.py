import torch
import numpy as np
import matplotlib.pyplot as plt

x = torch.arange(-3.0, 3.0, 0.1)
func = x**2 - 2 * torch.cos(x) - 5
predict = func + torch.empty_like(func).normal_(0, 0.5)

loss = torch.nn.MSELoss()
q = torch.mean((predict - func) ** 2)
print(q.item())
q_mse = loss(predict, func)
print(q_mse.item())

plt.plot(x, func, label="function")
plt.plot(x, predict, label="predict")
plt.legend()
plt.show()
