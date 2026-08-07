import torch

from subsurface_ml.neural_network import LithologyNeuralNetwork


def test_neural_network_output_shape():
    model = LithologyNeuralNetwork(
        input_size=10,
        num_classes=5,
    )

    x = torch.randn(4, 10)

    output = model(x)

    assert output.shape == (4, 5)