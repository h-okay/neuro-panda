import numpy as np
from src.network.neuron import Neuron


class Layer:
    def __init__(self, input_size, output_size, activation_function):
        self.neurons = [Neuron(activation_function) for _ in range(output_size)]
        self.input_size = input_size
        self.output_size = output_size
        self.activation_function = activation_function
        for neuron in self.neurons:
            neuron.weights = np.random.randn(
                input_size,
            )
            neuron.bias = np.random.randn(1)

    def forward(self, inputs):
        return np.array([neuron.forward(inputs) for neuron in self.neurons])

    def backward(self, errors):
        return np.array(
            [neuron.backward(error) for neuron, error in zip(self.neurons, errors)]
        )
