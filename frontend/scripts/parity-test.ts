import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { aiMove, type Weights } from "../src/ai/qnet.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = join(__dirname, "../..");
const weightsPath = join(__dirname, "../public/weights.json");
const weights = JSON.parse(readFileSync(weightsPath, "utf-8")) as Weights;

const board = [0, 0, 0, 0, 1, 0, 0, 0, 0];
const tsMove = aiMove(weights, board, -1);

const python = spawnSync(
  "python",
  [
    "-c",
    `
from model import QNet
import torch

def ai_move(model, board, player):
    x = torch.tensor([v * player for v in board], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q = model(x)[0]
    for i in range(9):
        if board[i] != 0:
            q[i] = -1e9
    return int(torch.argmax(q).item())

model = QNet()
model.load_state_dict(torch.load("../tictactoe_model.pth", map_location="cpu", weights_only=True))
model.eval()
print(ai_move(model, [0,0,0, 0,1,0, 0,0,0], -1))
`,
  ],
  { cwd: join(root, "tictactoe"), encoding: "utf-8" },
);

if (python.status !== 0) {
  console.error(python.stderr);
  process.exit(1);
}

const pyMove = Number(python.stdout.trim());

if (tsMove !== pyMove) {
  console.error(`Parity failed: TS=${tsMove}, Python=${pyMove}`);
  process.exit(1);
}

console.log(`Parity OK: both chose cell ${tsMove}`);
