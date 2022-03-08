from matplotlib.pyplot import axis
import numpy as np
D, N = 8,7
x = np.random.randn(1, D)
y = np.repeat(x, N, axis=0)

dy = np.random.randn(N, D)
dy.shape
dx = np.sum(dy, axis=0, keepdims=True) # 축 0을 모두 sum
dx.shape

