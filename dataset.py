import os
import json
from PIL import Image

from tqdm import tqdm

import torch
import torch.utils.data as data
import torchvision.transforms.v2 as tfs
from torchvision.datasets import ImageFolder

import torch.nn as nn
import torch.optim as optim


class DigitsDataset(data.Dataset):
    def __init__(self, path, train=True, transform=None):
        self.path = os.path.join(path, "train" if train else "test")
        self.transform = transform

        with open(os.path.join(path, "format.json"), "r") as fp:
            self.format = json.load(fp)
        self.length = 0
        self.files = []
        self.target = torch.eye(10)

        for _dir, _target in self.format.items():
            path = os.path.join(self.path, _dir)
            list_files = os.listdir(path)
            self.length += len(list_files)
            self.files.extend(
                map(lambda _x: (os.path.join(path, _x), _target), list_files)
            )

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        path_file, target = self.files[item]
        t = self.target[target]
        image = Image.open(path_file)

        if self.transform:
            image = self.transform(image).ravel().float() / 255.0

        return image, t


class DigitNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.functional.relu(x)
        x = self.layer2(x)
        return x


model = DigitNet(input_dim=28 * 28, hidden_dim=32, output_dim=10)

# to_tensor = tfs.ToImage()
# d_train = DigitsDataset(path="dataset", transform=to_tensor)
transform = tfs.Compose(
    [
        tfs.ToImage(),
        tfs.Grayscale(),
        tfs.ToDtype(torch.float32, scale=True),
        tfs.Lambda(lambda _img: _img.ravel()),
    ]
)
d_train = ImageFolder(root="dataset/train", transform=transform)

train_data = data.DataLoader(d_train, batch_size=32, shuffle=True)

# it = iter(train_data)
# image, target = next(it)
# print(image.shape)
# print(target.shape)

optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss()
epochs = 2
model.train()
for _ in range(epochs):
    train_tqdm = tqdm(train_data, desc="Training", leave=True)
    for image, target in train_tqdm:
        predict = model(image)
        loss = loss_func(predict, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

d_test = ImageFolder(root="dataset/test", transform=transform)
test_data = data.DataLoader(d_test, batch_size=500, shuffle=False)
q = 0


model.eval()
for x_test, y_test in test_data:
    with torch.no_grad():
        p = model(x_test)
        p = torch.argmax(p, dim=1)
        # y = torch.argmax(y_test, dim=1)
        q += torch.sum(p == y_test).item()

q = q / len(d_test)
print(q)
