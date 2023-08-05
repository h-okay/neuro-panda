from src.functions.activation import ActivationFunction
import numpy as np

class Tanh(ActivationFunction):
    def __call__(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1 - np.tanh(x) ** 2
