import torch

x = torch.tensor([0, 1, 4, -2, 5])
print(x.argmax(0))
