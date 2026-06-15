import torch


def act(x):
    return 0 if x < 0.5 else 1


def go(attr1, attr2, attr3):
    # переводим в тензор
    x = torch.tensor([attr1, attr2, attr3], dtype=torch.float32)
    # веса скрытого слоя
    w_h = torch.tensor([[0.3, 0.3, 0], [0.4, -0.5, 1.0]])
    # веса выходного слоя
    w_out = torch.tensor([-1.0, 1.0])

    # вычисляем сумму взвешенных входов
    z_h = torch.mv(w_h, x)
    print(z_h)
    # вычисляем активации
    u_h = torch.tensor([act(x) for x in z_h], dtype=torch.float32)
    print(u_h)
    # вычисляем сумму взвешенных выходов
    z_out = torch.dot(w_out, u_h)
    print(z_out)
    # вычисляем активацию на выходном слое
    y = act(z_out)
    # возвращаем результат
    print(y)
    return y


print(go(0, 0, 1))
