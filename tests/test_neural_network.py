import numpy as np
import pytest
from src.network.layer import Layer
from src.network.neural_network import NeuralNetwork
from src.network.neuron import Neuron
from src.functions.sigmoid import Sigmoid


@pytest.fixture
def sigmoid_activation():
    return Sigmoid()


def test_layer_forward(sigmoid_activation):
    activation_function = sigmoid_activation
    layer = Layer(input_size=2, output_size=2, activation_function=activation_function)
    layer.weights = np.array([[0.1, 0.2], [0.4, 0.5]])
    layer.bias = np.array([0.1, 0.2])
    inputs = np.array([[0.5, 0.6], [0.7, 0.8]])
    expected_output = activation_function(np.dot(inputs, layer.weights) + layer.bias)
    assert np.allclose(layer.forward(inputs), expected_output)


def test_neuron_forward(sigmoid_activation):
    activation_function = sigmoid_activation
    neuron = Neuron(activation_function)
    neuron.weights = np.array([0.1, 0.2])
    neuron.bias = np.array([0.1])
    inputs = np.array([[0.5, 0.6]])
    expected_output = activation_function(np.dot(inputs, neuron.weights) + neuron.bias)
    assert np.allclose(neuron.forward(inputs), expected_output)


def test_neural_network_forward():
    nn = NeuralNetwork(input_size=2, hidden_size=3, output_size=1)
    nn.hidden_layer.weights = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    nn.hidden_layer.bias = np.array([0.1, 0.2, 0.3])
    nn.output_layer.weights = np.array(
        [[0.7], [0.8], [0.9]]
    )  # Fix the shape of the weights
    nn.output_layer.bias = np.array([0.4])
    inputs = np.array([[0.5, 0.6], [0.7, 0.8]])
    hidden_output = nn.hidden_layer.forward(inputs)
    expected_output = nn.output_layer.activation_function(
        np.dot(hidden_output, nn.output_layer.weights) + nn.output_layer.bias
    )
    assert np.allclose(nn.forward(inputs), expected_output)


def test_neural_network_train():
    # Implement training tests here, if needed.
    pass
