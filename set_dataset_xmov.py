import json
import os
from random import choice, randint

import pygame

train_data = {"total": 10000, "dir": "train"}
test_data = {"total": 1000, "dir": "test"}
dir_out = "dataset_xmov"
file_format = "format.json"

CANVAS = 256
CHAR_H = 56
ROWS = 8
COLS = 12
# fraction of sprite size between neighbors (< 1.0 => overlap)
STEP_X_RATIO = 0.55
STEP_Y_RATIO = 0.45

if not os.path.exists(dir_out):
    os.mkdir(dir_out)
for split in ("train", "test"):
    path = os.path.join(dir_out, split)
    if not os.path.exists(path):
        os.mkdir(path)

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pygame.init()
pygame.display.set_mode((1, 1))

xmov_src = pygame.image.load("images/xmov.png").convert_alpha()
not_xmov_src = pygame.image.load("images/not_xmov.png").convert_alpha()

scale = CHAR_H / xmov_src.get_height()
char_w = max(1, int(xmov_src.get_width() * scale))
char_h = CHAR_H

xmov = pygame.transform.smoothscale(xmov_src, (char_w, char_h))
not_xmov = pygame.transform.smoothscale(not_xmov_src, (char_w, char_h))
xmov.set_colorkey((0, 0, 0))
not_xmov.set_colorkey((0, 0, 0))

step_x = char_w * STEP_X_RATIO
step_y = char_h * STEP_Y_RATIO
# bleed past all edges so the frame is a crop of a dense crowd
crowd_w = (COLS - 1) * step_x + char_w
crowd_h = (ROWS - 1) * step_y + char_h
origin_x = (CANVAS - crowd_w) / 2
origin_y = (CANVAS - crowd_h) / 2


def make_positions():
    positions = []
    for row in range(ROWS):
        y = origin_y + row * step_y
        x_shift = step_x * 0.5 if row % 2 else 0.0
        for col in range(COLS):
            x = origin_x + col * step_x + x_shift
            jx = randint(-4, 4)
            jy = randint(-3, 3)
            px = int(x + jx)
            py = int(y + jy)
            positions.append((px, py))
    return positions


def random_bg():
    # soft solid backgrounds — characters stay readable
    palette = [
        (40, 44, 52),
        (30, 60, 90),
        (55, 70, 50),
        (70, 50, 60),
        (90, 90, 95),
        (25, 25, 35),
    ]
    color = choice(palette)
    # slight per-channel noise
    color = tuple(max(0, min(255, c + randint(-8, 8))) for c in color)
    surf = pygame.Surface((CANVAS, CANVAS))
    surf.fill(color)
    return surf


for info in (train_data, test_data):
    xmov_coords = {}

    for i in range(1, info["total"] + 1):
        file_out = f"xmov_{i}.png"
        im = random_bg()
        positions = make_positions()

        xmov_idx = randint(0, len(positions) - 1)
        # draw back rows first, front rows last
        order = list(range(len(positions)))
        # optional: shuffle within same row band is unnecessary; draw top->bottom
        for idx in order:
            sprite = xmov if idx == xmov_idx else not_xmov
            im.blit(sprite, positions[idx])

        x, y = positions[xmov_idx]
        # center of the character — same convention as sun dataset
        cx = max(0, min(CANVAS - 1, x + char_w // 2))
        cy = max(0, min(CANVAS - 1, y + char_h // 2))
        xmov_coords[file_out] = (cx, cy)

        pygame.image.save(im, os.path.join(dir_out, info["dir"], file_out))

    with open(os.path.join(dir_out, info["dir"], file_format), "w") as fp:
        json.dump(xmov_coords, fp)

print(f"done: {dir_out}/train + {dir_out}/test, char={char_w}x{char_h}")
