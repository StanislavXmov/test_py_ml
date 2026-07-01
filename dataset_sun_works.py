from PIL import Image
import json
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms.v2 as tfs


def resolve_device(device_name: str | None) -> torch.device:
    if device_name is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


device = resolve_device(None)

model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding="same"),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 8, 3, padding="same"),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(8, 4, 3, padding="same"),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(4096, 128),
    nn.ReLU(),
    nn.Linear(128, 2),
)
model.to(device)

path = "dataset_reg/test/"
num_img = 50

st = torch.load("model_sun.tar", weights_only=False)
model.load_state_dict(st)

with open(os.path.join(path, "format.json"), "r") as fp:
    format = json.load(fp)

transforms = tfs.Compose([tfs.ToImage(), tfs.ToDtype(torch.float32, scale=True)])
img = Image.open(os.path.join(path, f"sun_reg_{num_img}.png")).convert("RGB")
img_t = transforms(img).unsqueeze(0)

model.eval()
predict = model(img_t.to(device, non_blocking=True))
print(predict)
print(tuple(format.values())[num_img - 1])
p = predict.detach().cpu().squeeze().numpy()

plt.imshow(img)
plt.scatter(p[0], p[1], s=20, c="r")
plt.show()
