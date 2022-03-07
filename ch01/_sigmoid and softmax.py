import numpy as np

W1 = np.random.randn(2, 4)
b1 = np.random.randn(4)
x = np.random.randn(10,2)
h = np.matmul(x, W1) +b1

def sigmoid(x):
    return 1 / (1+np.exp(-x))

a = sigmoid(h)

def softmax(x):
    e_x = np.exp(x)
    sum_e_x = np.sum(e_x, axis=1)
    y = e_x / np.repeat(sum_e_x.reshape(10,1), 4, axis=1)

    return y

np.sum(np.exp(h), axis=1).shape

np.sum(softmax(a), axis=1)