from typing import Callable, List

import numpy as np


class Neuron:
    def __init__(self, weights: List[float], bias: float) -> None:
        self.weights = weights
        self.bias = bias

    def feedforward(
        self, inputs: List[float], activation_function: Callable[[float], float]
    ) -> float:
        total = np.dot(self.weights, inputs) + self.bias
        return activation_function(total)
