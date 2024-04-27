import numpy as np
import torch

z = torch.tensor([0.0, 1.0, 5.0, 1.0, 1.0, 1.0])

x1 = z[:3]
m1 = z[3:]

dist = torch.distributions.Normal(x1, m1)

print(dist.sample())

