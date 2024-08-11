# the network is a wrapper for layers
# the network's job is to call each layer's forward method on the inputs and pass the output to the next layer
# the network returns back a list of values manipulated by the layers

from functions.loss import calculate_mean_loss
from modules.layer import Layer


class Network:
    def __init__(self, layers: list[Layer]):
        self.layers = layers

    def _forward(self, inputs: list[float]) -> list[float]:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def train(
        self,
        inputs: list[float],
        epochs: int,
        expected: list[float],
        learning_rate: float,
    ):
        for epoch in range(epochs):
            predictions = self._forward(inputs)
            loss = calculate_mean_loss(predictions, expected)
            print(f"Epoch: {epoch}, Loss: {loss}, Prediction: {predictions}")

            # Backpropagation
            self.backpropagation(expected, learning_rate)

    def backpropagation(self, expected: list[float], learning_rate: float):
        for layer in reversed(self.layers):
            expected = layer.backpropagation(expected, learning_rate)
        return expected
