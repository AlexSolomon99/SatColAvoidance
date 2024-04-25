import numpy as np

x = np.array([1, 2, 3, 4, 5])
print(x)
print(x.shape)

print(np.where(x == 4)[0][0])
