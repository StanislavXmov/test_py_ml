import torch
import torch.nn as nn
import torch.optim as optim

from random import randint


class NeuroTest(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, inp):
        x = self.layer1(inp)
        x = torch.tanh(x)
        x = self.layer2(x)
        x = torch.tanh(x)
        return x


# def forward(inp, l1: nn.Linear, l2: nn.Linear):
#     u1 = l1(inp)
#     s1 = torch.tanh(u1)

#     u2 = l2(s1)
#     s2 = torch.tanh(u2)
#     return s2


# layer1 = nn.Linear(in_features=3, out_features=2)
# layer2 = nn.Linear(2, 1)

# layer1.weight.data = torch.tensor(
#     [[0.7402, 0.6008, -1.3340], [0.2098, 0.4537, -0.7692]]
# )
# layer1.bias.data = torch.tensor([0.5505, 0.3719])

# layer2.weight.data = torch.tensor([[-2.0719, -0.9485]])
# layer2.bias.data = torch.tensor([-0.1461])

# x = torch.tensor([1, -1, 1], dtype=torch.float32)
# y = forward(x, layer1, layer2)
# print(y.data)

model = NeuroTest(input_dim=3, hidden_dim=2, output_dim=1)

x_train = torch.FloatTensor(
    [
        (-1, -1, -1),
        (-1, -1, 1),
        (-1, 1, -1),
        (-1, 1, 1),
        (1, -1, -1),
        (1, -1, 1),
        (1, 1, -1),
        (1, 1, 1),
    ]
)
y_train = torch.FloatTensor([-1, 1, -1, 1, -1, 1, -1, -1])
total = len(y_train)

optimizer = optim.RMSprop(model.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()

model.train()
for _ in range(1000):
    k = randint(0, total - 1)
    x = x_train[k]
    y = model(x)
    y = y.squeeze()
    loss = loss_func(y, y_train[k])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


model.eval()
for x, d in zip(x_train, y_train):
    with torch.no_grad():
        y = model(x)
        print(f"Выходное значение НС: {y.data} => {d}")
