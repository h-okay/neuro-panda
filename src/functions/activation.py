import numpy as np


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def relu(x: float) -> float:
    return max(0, x)


def tanh(x: float) -> float:
    return np.tanh(x)
