import random
import torch
import torch.nn as nn
import torch.optim as optim

from model import QNet


MODEL_PATH = "tictactoe_model.pth"


def winner(board):
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


def available_moves(board: list[int]):
    return [i for i, v in enumerate(board) if v == 0]


def encode(board: list[int], player: int):
    return torch.tensor([v * player for v in board], dtype=torch.float32)


def choose_move(model: QNet, board: list[int], player: int, eps: float):
    moves = available_moves(board)

    if random.random() < eps:
        return random.choice(moves)

    with torch.no_grad():
        x = encode(board, player).unsqueeze(0)
        q = model(x)[0]

    for i in range(9):
        if board[i] != 0:
            q[i] = -1e9

    return int(torch.argmax(q).item())


def train():
    model = QNet()
    target_model = QNet()
    target_model.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    gamma = 0.95
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.9995

    replay = []
    replay_limit = 10000
    batch_size = 64

    episodes = 20000

    for episode in range(episodes):
        board = [0] * 9
        player = 1

        while True:
            state = board.copy()
            move = choose_move(model, board, player, epsilon)

            board[move] = player
            result = winner(board)

            if result is not None:
                if result == player:
                    reward = 1.0
                elif result == 0:
                    reward = 0.2
                else:
                    reward = -1.0

                replay.append((state, player, move, reward, board.copy(), True))
                break

            replay.append((state, player, move, 0.0, board.copy(), False))

            player *= -1

            if len(replay) > replay_limit:
                replay.pop(0)

            if len(replay) >= batch_size:
                batch = random.sample(replay, batch_size)

                states, players, moves, rewards, next_states, dones = zip(*batch)

                states_tensor = torch.stack(
                    [encode(s, p) for s, p in zip(states, players)]
                )

                next_states_tensor = torch.stack(
                    [encode(s, -p) for s, p in zip(next_states, players)]
                )

                moves_tensor = torch.tensor(moves, dtype=torch.long)
                rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
                dones_tensor = torch.tensor(dones, dtype=torch.bool)

                q_values = model(states_tensor)
                q_selected = q_values.gather(1, moves_tensor.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_q = target_model(next_states_tensor)

                    for row, next_state in enumerate(next_states):
                        for i in range(9):
                            if next_state[i] != 0:
                                next_q[row, i] = -1e9

                    next_max_q = next_q.max(dim=1).values

                    target = rewards_tensor - gamma * next_max_q
                    target[dones_tensor] = rewards_tensor[dones_tensor]

                loss = loss_fn(q_selected, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if episode % 500 == 0:
            target_model.load_state_dict(model.state_dict())

        if episode % 1000 == 0:
            print(f"Episode {episode}, " f"epsilon={epsilon:.3f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
