import numpy as np
from typing import List
from src.network.neuron import Neuron


class NeuralNetwork:
    def __init__(
        self,
        weights: List[float],
        bias: float,
        hidden_layer: List[Neuron],
        output_layer: Neuron,
    ) -> None:
        self.weights = weights
        self.bias = bias
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer

    def feedforward(self, x):
        network = [layer.feedforward(x) for layer in self.hidden_layer]
        return self.output_layer.feedforward(np.array(network))
