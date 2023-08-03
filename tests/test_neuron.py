from src.functions.activation import relu, sigmoid, tanh
from src.network.neuron import Neuron


def test_feedforward_sigmoid():
    neuron = Neuron([0.5, 0.5], -1, sigmoid)
    input_data = [2, 3]
    expected_output = 0.8175744761936437
    assert expected_output == neuron.feedforward(input_data)


def test_feedforward_relu():
    neuron = Neuron([1, -1], 0, relu)
    input_data = [10, 5]
    expected_output = 5
    assert expected_output == neuron.feedforward(input_data)


def test_feedforward_tanh():
    neuron = Neuron([0.2, 0.3], 0.1, tanh)
    input_data = [1, 1]
    expected_output = 0.5370495669980352
    assert expected_output == neuron.feedforward(input_data)
