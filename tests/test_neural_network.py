from src.network.neural_network import NeuralNetwork
from src.network.neuron import Neuron
from src.functions.activation import sigmoid, relu, tanh


def test_feedforward_sigmoid():
    hidden_layer = [Neuron([0.5, 0.5], -1, sigmoid), Neuron([1, -1], 0, sigmoid)]
    output_layer = Neuron([0.2, 0.3], 0.1, sigmoid)
    network_builder = NeuralNetwork([1, 1], 0, hidden_layer, output_layer)
    input_data = [2, 3]
    expected_output = 0.5852097408844837
    expected_output, network_builder.feedforward(input_data)


def test_feedforward_relu():
    hidden_layer = [Neuron([0.5, 0.5], -1, relu), Neuron([1, -1], 0, relu)]
    output_layer = Neuron([0.2, 0.3], 0.1, relu)
    network_builder = NeuralNetwork([1, 1], 0, hidden_layer, output_layer)
    input_data = [2, 3]
    expected_output = 1.3
    expected_output, network_builder.feedforward(input_data)


def test_feedforward_tanh():
    hidden_layer = [Neuron([0.5, 0.5], -1, tanh), Neuron([1, -1], 0, tanh)]
    output_layer = Neuron([0.2, 0.3], 0.1, tanh)
    network_builder = NeuralNetwork([1, 1], 0, hidden_layer, output_layer)
    input_data = [2, 3]
    expected_output = 0.6913526512322059
    expected_output, network_builder.feedforward(input_data)
