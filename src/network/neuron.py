import numpy as np


class Neuron:
    def __init__(self, activation_function):
        self.activation_function = activation_function
        self.inputs = None
        self.weights = None
        self.bias = None
        self.output = None

    def forward(self, inputs):
        self.inputs = inputs
        self.output = self.activation_function(np.dot(inputs, self.weights) + self.bias)
        return self.output

    def backward(self, error):
        return error * self.activation_function.derivative(
            np.dot(self.inputs, self.weights) + self.bias
        )
