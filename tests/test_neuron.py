import numpy as np

from src.functions.sigmoid import sigmoid
from src.network.neuron import Neuron


def test_feedforward_with_zero_weights_and_zero_bias():
    neuron = Neuron(np.zeros(3), 0.0)
    inputs = np.array([1.0, 2.0, 3.0])
    result = neuron.feedforward(inputs)
    assert result == sigmoid(0.0)


def test_feedforward_with_nonzero_weights_and_zero_bias():
    neuron = Neuron(np.array([0.5, 0.5]), 0.0)
    inputs = np.array([1.0, 1.0])
    result = neuron.feedforward(inputs)
    assert result == sigmoid(1)


def test_feedforward_with_zero_weights_and_nonzero_bias():
    neuron = Neuron(np.zeros(2), 0.5)
    inputs = np.array([2.0, 3.0])
    result = neuron.feedforward(inputs)
    assert result == sigmoid(0.5)


def test_feedforward_with_nonzero_weights_and_nonzero_bias():
    neuron = Neuron(np.array([2.0, 3.0]), 1.0)
    inputs = np.array([0.5, 1.0])
    result = neuron.feedforward(inputs)
    assert result == sigmoid(5)
