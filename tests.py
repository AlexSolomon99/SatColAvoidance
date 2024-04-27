import numpy as np
import torch

x = torch.tensor([2, 3, 4, 5])
scores_list = torch.tensor([])

for idx, elem in enumerate(x):
    new_num = elem.item() + 5
    new_num_tensor = torch.Tensor([new_num])
    scores_list = torch.cat((scores_list, new_num_tensor))

print(x)
print(scores_list)
