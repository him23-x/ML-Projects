import numpy as np
from scipy.stats import gamma

# inputs
x = np.array([1.2, 0.8, 1.5, 2.0])  
beta = 2
a = 1
b = 1


n = len(x)
shape = n + a
rate = np.sum(x**beta) + b
scale = 1 / rate  


L = gamma.ppf(0.025, a=shape, scale=scale)
U = gamma.ppf(0.975, a=shape, scale=scale)

print("Lower bound L =", L)
print("Upper bound U =", U)