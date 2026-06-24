import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim


class FuncDataset(data.Dataset):
    def __init__(self):
        _x = torch.arange(-5, 5, 0.1)
        self.data = _x
        self.target = torch.sin(2 * _x) + 0.2 * torch.cos(10 * _x) + 0.1 * _x**2
        self.length = len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.target[item]

    def __len__(self):
        return self.length


class FuncModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(3, 1)

    def forward(self, x):
        xx = torch.empty(x.size(0), 3)
        xx[:, 0] = x
        xx[:, 1] = x**2
        xx[:, 2] = x**3
        y = self.layer1(xx)
        return y


torch.manual_seed(1)

model = FuncModel()
model.train()

epochs = 20
batch_size = 8

d_train = FuncDataset()
train_data = data.DataLoader(d_train, batch_size=batch_size, shuffle=True)

optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_func = nn.MSELoss()

for _e in range(epochs):
    for x_train, y_train in train_data:
        predict = model(x_train)
        loss = loss_func(predict, y_train.unsqueeze(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.eval()
predict = model(d_train.data)
Q = loss_func(predict, d_train.target.unsqueeze(-1)).item()
print(Q)
