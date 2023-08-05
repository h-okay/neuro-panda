from src.network.layer import Layer
from src.functions.sigmoid import Sigmoid
from src.functions.relu import ReLU
from src.functions.tanh import Tanh
import numpy as np


class NeuralNetwork:
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        hidden_activation="sigmoid",
        output_activation="sigmoid",
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.hidden_layer = Layer(
            input_size, hidden_size, self._get_activation_function(hidden_activation)
        )
        self.output_layer = Layer(
            hidden_size, output_size, self._get_activation_function(output_activation)
        )

        self.hidden_layer.neurons[0].weights = np.random.randn(input_size)
        self.hidden_layer.neurons[1].weights = np.random.randn(input_size)
        self.hidden_layer.neurons[2].weights = np.random.randn(input_size)

        self.output_layer.neurons[0].weights = np.random.randn(hidden_size)
        self.output_layer.neurons[0].bias = np.random.randn()

    def _get_activation_function(self, activation):
        if activation == "sigmoid":
            return Sigmoid()
        elif activation == "relu":
            return ReLU()
        elif activation == "tanh":
            return Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, inputs):
        hidden_output = self.hidden_layer.forward(inputs)
        return self.output_layer.forward(hidden_output)

    def backward(self, inputs, targets):
        output_errors = targets - self.forward(inputs)
        hidden_errors = self.output_layer.backward(output_errors)
        return hidden_errors

    def train(self, inputs, targets, learning_rate=0.1, epochs=1000):
        for epoch in range(epochs):
            hidden_output = self.hidden_layer.forward(inputs)
            output = self.output_layer.forward(hidden_output)

            output_errors = targets - output
            output_delta = (
                output_errors * self.output_layer.activation_function.derivative(output)
            )

            hidden_errors = self.output_layer.backward(output_errors)
            hidden_delta = (
                hidden_errors
                * self.hidden_layer.activation_function.derivative(hidden_output)
            )

            self.output_layer.weights += (
                np.dot(hidden_output.T, output_delta) * learning_rate
            )
            self.output_layer.bias += (
                np.sum(output_delta, axis=0, keepdims=True) * learning_rate
            )

            self.hidden_layer.weights += np.dot(inputs.T, hidden_delta) * learning_rate
            self.hidden_layer.bias += (
                np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate
            )

    def loss(self, y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)
