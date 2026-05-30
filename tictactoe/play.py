from model import QNet
import torch


def encode(board: list[int], player: int):
    return torch.tensor([v * player for v in board], dtype=torch.float32)


def winner(board: list[int]):
    lines = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],
        [0, 4, 8],
        [2, 4, 6],
    ]

    for a, b, c in lines:
        s = board[a] + board[b] + board[c]

        if s == 3:
            return 1

        if s == -3:
            return -1

    if 0 not in board:
        return 0

    return None


def print_board(board: list[int]):
    chars = []

    for v in board:
        if v == 1:
            chars.append("X")
        elif v == -1:
            chars.append("O")
        else:
            chars.append(".")

    print()
    print(chars[0], chars[1], chars[2])
    print(chars[3], chars[4], chars[5])
    print(chars[6], chars[7], chars[8])
    print()


def ai_move(model: QNet, board: list[int], player: int):
    x = encode(board, player).unsqueeze(0)

    with torch.no_grad():
        q = model(x)[0]

    for i in range(9):
        if board[i] != 0:
            q[i] = -1e9

    return int(torch.argmax(q).item())


# Загружаем модель
model = QNet()
model.load_state_dict(torch.load("tictactoe_model.pth", weights_only=True))
model.eval()


# 0 1 2
# 3 4 5
# 6 7 8

# человек = X
# ИИ = O
human = 1
ai = -1

board = [0] * 9
player = human

# while True:
#     print_board(board)

#     result = winner(board)

#     if result is not None:
#         if result == human:
#             print("Вы победили!")
#         elif result == ai:
#             print("Победил ИИ!")
#         else:
#             print("Ничья!")
#         break

#     if player == human:
#         while True:
#             move = int(input("Ход (0-8): "))

#             if 0 <= move <= 8 and board[move] == 0:
#                 board[move] = human
#                 break

#             print("Некорректный ход")
#     else:
#         move = ai_move(model, board, ai)
#         print(f"ИИ ходит в клетку {move}")
#         board[move] = ai

#     player *= -1

board[4] = human
print(ai_move(model, board, ai))
