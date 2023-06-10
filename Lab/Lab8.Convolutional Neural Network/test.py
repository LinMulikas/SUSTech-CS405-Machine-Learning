import torch

a = torch.tensor([], dtype=torch.int)
b = torch.tensor([4, 5, 6])

print(torch.cat((a, b), dim=0))