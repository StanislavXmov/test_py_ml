import json
from pathlib import Path

import torch

from model import QNet

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "tictactoe_model.pth"
OUTPUT_PATH = ROOT / "frontend" / "public" / "weights.json"


def export_weights():
    model = QNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()

    weights = {
        "w1": model.net[0].weight.tolist(),
        "b1": model.net[0].bias.tolist(),
        "w2": model.net[2].weight.tolist(),
        "b2": model.net[2].bias.tolist(),
        "w3": model.net[4].weight.tolist(),
        "b3": model.net[4].bias.tolist(),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(weights, f)

    print(f"Weights exported to {OUTPUT_PATH}")


if __name__ == "__main__":
    export_weights()
