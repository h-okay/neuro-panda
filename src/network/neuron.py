import numpy as np

from src.functions.sigmoid import sigmoid


class Neuron:
    def __init__(self, weights, bias) -> None:
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)
