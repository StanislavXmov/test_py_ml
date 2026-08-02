from PIL import Image
import json
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
import torch.nn as nn
import torchvision.transforms.v2 as tfs

# размер спрайта как в set_dataset_xmov.py
CHAR_W, CHAR_H = 31, 56


def resolve_device(device_name: str | None) -> torch.device:
    if device_name is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


device = resolve_device(None)

# 256 -> 128 -> 64 -> 32; 64*32*32 = 65536
model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding="same"),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding="same"),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(64, 64, 3, padding="same"),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(65536, 256),
    nn.ReLU(),
    nn.Linear(256, 2),
)
model.to(device)

path = "dataset_xmov/test/"
num_img = 664

st = torch.load("model_xmov.tar", weights_only=False)
model.load_state_dict(st)

with open(os.path.join(path, "format.json"), "r") as fp:
    format = json.load(fp)

transforms = tfs.Compose([tfs.ToImage(), tfs.ToDtype(torch.float32, scale=True)])
img = Image.open(os.path.join(path, f"xmov_{num_img}.png")).convert("RGB")
img_t = transforms(img).unsqueeze(0)

model.eval()
predict = model(img_t.to(device, non_blocking=True))
print(predict)
print(tuple(format.values())[num_img - 1])
p = predict.detach().cpu().squeeze().numpy() * 255.0
x0 = p[0] - CHAR_W / 2
y0 = p[1] - CHAR_H / 2

plt.imshow(img)
plt.gca().add_patch(
    Rectangle((x0, y0), CHAR_W, CHAR_H, fill=False, edgecolor="lime", linewidth=2)
)
plt.show()
