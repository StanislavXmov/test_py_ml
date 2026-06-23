import os
import json
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms.v2 as tfs


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


to_tensor = tfs.ToImage()
d_train = DigitsDataset(path="dataset", transform=to_tensor)
train_data = data.DataLoader(d_train, batch_size=32, shuffle=True)

# it = iter(train_data)
# image, target = next(it)
# print(image.shape)
# print(target.shape)
