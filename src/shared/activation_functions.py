import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True)) # stabilisasi utk mencegah nilai NaN/Inf
    return e_x / np.sum(e_x, axis=-1, keepdims=True)