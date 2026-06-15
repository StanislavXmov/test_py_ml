import torch


def act(x):
    return 0 if x <= 0 else 1


def xor(x1, x2):
    w_hidden = torch.FloatTensor([[1, 1, -1.5], [1, 1, -0.5]])
    w_out = torch.FloatTensor([-1, 1, -0.5])
    data_x = [x1, x2]
    x = torch.FloatTensor(data_x + [1])
    z_hidden = torch.matmul(w_hidden, x)
    u_hidden = torch.FloatTensor([act(x) for x in z_hidden] + [1])
    z_out = torch.dot(w_out, u_hidden)
    y = act(z_out)
    return y


print(xor(0, 0))
print(xor(0, 1))
print(xor(1, 0))
print(xor(1, 1))
