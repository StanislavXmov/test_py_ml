import argparse
import json
import os
import time

from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms.v2 as tfs
from torchvision.datasets import ImageFolder


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


def resolve_device(device_name: str | None) -> torch.device:
    if device_name is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


class FlattenTransform:
    def __call__(self, img):
        return img.ravel()


def make_transform():
    return tfs.Compose(
        [
            tfs.ToImage(),
            tfs.Grayscale(),
            tfs.ToDtype(torch.float32, scale=True),
            FlattenTransform(),
        ]
    )


def make_dataloader(dataset, batch_size, shuffle, num_workers):
    use_cuda = torch.cuda.is_available()
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
    )


def train(
    device_name: str | None = None,
    epochs: int = 2,
    batch_size: int = 256,
    num_workers: int = 4,
    dataset_root: str = "dataset",
):
    device = resolve_device(device_name)
    print(f"Using {device}")
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))

    transform = make_transform()
    d_train = ImageFolder(root=os.path.join(dataset_root, "train"), transform=transform)
    train_loader = make_dataloader(
        d_train, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    model = DigitNet(input_dim=28 * 28, hidden_dim=32, output_dim=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_func = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        train_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=True)
        for image, target in train_tqdm:
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            predict = model(image)
            loss = loss_func(predict, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        elapsed = time.perf_counter() - epoch_start
        print(f"Epoch {epoch + 1} finished in {elapsed:.2f}s")

    d_test = ImageFolder(root=os.path.join(dataset_root, "test"), transform=transform)
    test_loader = make_dataloader(
        d_test, batch_size=500, shuffle=False, num_workers=num_workers
    )

    correct = 0
    model.eval()
    for x_test, y_test in test_loader:
        x_test = x_test.to(device, non_blocking=True)
        y_test = y_test.to(device, non_blocking=True)
        with torch.no_grad():
            p = model(x_test)
            p = torch.argmax(p, dim=1)
            correct += torch.sum(p == y_test).item()

    accuracy = correct / len(d_test)
    print(f"Test accuracy: {accuracy:.4f}")
    return accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Train DigitNet on image folder dataset"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="PyTorch device (default: cuda if available, else cpu)",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dataset-root", default="dataset")
    args = parser.parse_args()

    train(
        device_name=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dataset_root=args.dataset_root,
    )


if __name__ == "__main__":
    main()
