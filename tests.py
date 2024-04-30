import numpy as np
import torch

log_probs = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # Shape: (batch_size, 3)
rewards = torch.tensor([10.0, 0.1])

print(log_probs * rewards.unsqueeze(1))
print(torch.sum(log_probs * rewards.unsqueeze(1), dim=1))

loss = torch.mean(torch.sum(log_probs * rewards.unsqueeze(1), dim=1))

print(loss)
