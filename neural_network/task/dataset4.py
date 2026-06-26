import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim


class WineDataset(data.Dataset):
    def __init__(self):
        self.data = _global_var_data_x  # тензор размерностью (178, 13), тип float32
        self.target = (
            _global_var_target  # тензор размерностью (178, ), тип int64 (long)
        )

        self.length = len(self.data)
        self.categories = ["class_0", "class_1", "class_2"]  # названия классов

    def __getitem__(self, item):
        return self.data[item], self.target[item]

    def __len__(self):
        return self.length


class WineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(13, 32)
        self.layer2 = nn.Linear(32, 16)
        self.layer3 = nn.Linear(16, 3)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.layer3(x)
        return x


model = WineModel()
model.train()

epochs = 20
batch_size = 16

d_train = WineDataset()
train_data = data.DataLoader(d_train, batch_size, shuffle=True)

optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss()


for _ in range(epochs):
    for x_train, y_train in train_data:
        predict = model(x_train)
        loss = loss_func(predict, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.eval()
predict = model(d_train.data)
p = torch.argmax(predict, dim=1)
Q = torch.mean((p == d_train.target).float()).item()
