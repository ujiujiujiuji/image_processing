import numpy as np

def sigmoid(t):
    T = t * -1
    a = 1 / (1 + np.exp(T))
    return a