from src.functions.activation import relu, sigmoid, tanh


def test_sigmoid():
    assert sigmoid(0) == 0.5
    assert sigmoid(10) == 0.9999546021312976
    assert sigmoid(-10) == 4.5397868702434395e-05
    assert sigmoid(1000) == 1.0
    assert sigmoid(-1000) == 0.0


def test_relu():
    assert relu(0) == 0
    assert relu(10) == 10
    assert relu(-10) == 0
    assert relu(1000) == 1000
    assert relu(-1000) == 0


def test_tanh():
    assert tanh(0) == 0.0
    assert tanh(10) == 0.9999999958776927
    assert tanh(-10) == -0.9999999958776927
    assert tanh(1000) == 1.0
    assert tanh(-1000) == -1.0
