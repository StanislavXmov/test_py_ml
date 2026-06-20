import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

x_train = torch.tensor(
    [
        (5.8, 1.2),
        (5.6, 1.5),
        (6.5, 1.5),
        (6.1, 1.3),
        (6.4, 1.3),
        (7.7, 2.0),
        (6.0, 1.8),
        (5.6, 1.3),
        (6.0, 1.6),
        (5.8, 1.9),
        (5.7, 2.0),
        (6.3, 1.5),
        (6.2, 1.8),
        (7.7, 2.3),
        (5.8, 1.2),
        (6.3, 1.8),
        (6.0, 1.0),
        (6.2, 1.3),
        (5.7, 1.3),
        (6.3, 1.9),
        (6.7, 2.5),
        (5.5, 1.2),
        (4.9, 1.0),
        (6.1, 1.4),
        (6.0, 1.6),
        (7.2, 2.5),
        (7.3, 1.8),
        (6.6, 1.4),
        (5.6, 2.0),
        (5.5, 1.0),
        (6.4, 2.2),
        (5.6, 1.3),
        (6.6, 1.3),
        (6.9, 2.1),
        (6.8, 2.1),
        (5.7, 1.3),
        (7.0, 1.4),
        (6.1, 1.4),
        (6.1, 1.8),
        (6.7, 1.7),
        (6.0, 1.5),
        (6.5, 1.8),
        (6.4, 1.5),
        (6.9, 1.5),
        (5.6, 1.3),
        (6.7, 1.4),
        (5.8, 1.9),
        (6.3, 1.3),
        (6.7, 2.1),
        (6.2, 2.3),
        (6.3, 2.4),
        (6.7, 1.8),
        (6.4, 2.3),
        (6.2, 1.5),
        (6.1, 1.4),
        (7.1, 2.1),
        (5.7, 1.0),
        (6.8, 1.4),
        (6.8, 2.3),
        (5.1, 1.1),
        (4.9, 1.7),
        (5.9, 1.8),
        (7.4, 1.9),
        (6.5, 2.0),
        (6.7, 1.5),
        (6.5, 2.0),
        (5.8, 1.0),
        (6.4, 2.1),
        (7.6, 2.1),
        (5.8, 2.4),
        (7.7, 2.2),
        (6.3, 1.5),
        (5.8, 1.0),
        (6.3, 1.6),
        (7.7, 2.3),
        (6.4, 1.9),
        (6.5, 2.2),
        (5.7, 1.2),
        (6.9, 2.3),
        (5.7, 1.3),
        (6.1, 1.2),
        (5.4, 1.5),
        (5.2, 1.4),
        (6.7, 2.3),
        (7.9, 2.0),
        (5.6, 1.1),
        (7.2, 1.8),
        (5.5, 1.3),
        (7.2, 1.6),
        (6.3, 2.5),
        (6.3, 1.8),
        (6.7, 2.4),
        (5.0, 1.0),
        (6.4, 1.8),
        (6.9, 2.3),
        (5.5, 1.3),
        (5.5, 1.1),
        (5.9, 1.5),
        (6.0, 1.5),
        (5.9, 1.8),
    ]
)

y_train = torch.tensor(
    [
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        1,
        1,
        0,
        1,
        1,
        0,
        1,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        1,
        0,
        1,
        0,
        0,
        1,
        1,
        0,
        0,
        1,
        1,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        1,
        0,
        1,
        1,
        0,
        1,
        0,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        1,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        1,
        0,
        1,
        1,
        1,
        1,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        1,
    ]
)

y_train = y_train.float()

total = len(y_train)
n_features = 3
lr = 0.1
N = 500

np.random.seed(1)

x = torch.ones(total, 3)
x[:, 1:3] = x_train
w = torch.empty(n_features).uniform_(-1e-5, 1e-5)
w.requires_grad_(True)
loss_func = torch.nn.BCEWithLogitsLoss()
optimizer = optim.Adam(params=[w], lr=lr)

for _ in range(N):
    k = np.random.randint(0, total)
    predict = x[k] @ w
    y = loss_func(predict, y_train[k])
    y.backward()
    optimizer.step()
    optimizer.zero_grad()

q = torch.mean((torch.sign(x @ w) == (y_train * 2 - 1)).float())

mask_0 = y_train == 0
mask_1 = y_train == 1

plt.figure(figsize=(8, 6))
plt.scatter(
    x_train[mask_0, 0],
    x_train[mask_0, 1],
    c="tab:blue",
    label="Класс 0",
    alpha=0.7,
)
plt.scatter(
    x_train[mask_1, 0],
    x_train[mask_1, 1],
    c="tab:orange",
    label="Класс 1",
    alpha=0.7,
)

with torch.no_grad():
    x1_line = torch.linspace(x_train[:, 0].min(), x_train[:, 0].max(), 100)
    x2_line = -(w[0] + w[1] * x1_line) / w[2]
plt.plot(x1_line.numpy(), x2_line.numpy(), "k", label="Решающая граница")
plt.xlabel("Признак 1")
plt.ylabel("Признак 2")
plt.title("Обучающая выборка")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
