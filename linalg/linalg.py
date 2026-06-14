import torch

a = torch.FloatTensor([[1, 2, 3], [1, 4, 9], [1, 8, 27]])
y = torch.FloatTensor([10, 20, 30])

rank = torch.linalg.matrix_rank(a)
solve = torch.linalg.solve(a, y)
print(solve)

inv_a = torch.linalg.inv(a)
x = torch.mv(inv_a, y)
print(x)
