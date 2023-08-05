from src.functions.activation import ActivationFunction
import numpy as np

class Sigmoid(ActivationFunction):
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        sigmoid = self(x)
        return sigmoid * (1 - sigmoid)
