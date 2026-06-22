import torch
import torch.nn as nn
import torch.utils.data as data


class TestDataset(data.Dataset):
    def __init__(self):
        self.coord_x = torch.arange(-5, 5, 0.1)
        self.func = (
            2 * torch.exp(-self.coord_x / 2) - 0.2 * torch.sin(self.coord_x / 10) - 5
        )
        self.length = len(self.coord_x)

    def __getitem__(self, index):
        return self.coord_x[index], self.func[index]

    def __len__(self):
        return self.length


dataset = TestDataset()
x13, y13 = dataset[13]
total = len(dataset)
print(x13, y13, total)
