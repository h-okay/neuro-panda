from typing import Callable, List

import numpy as np


class Neuron:
    def __init__(
        self,
        weights: List[float],
        bias: float,
        activation_function: Callable[[float], float],
    ) -> None:
        self.weights = weights
        self.bias = bias
        self.activation_function = activation_function

    def feedforward(
        self,
        inputs: List[float],
    ) -> float:
        total = np.dot(self.weights, inputs) + self.bias
        return self.activation_function(total)
