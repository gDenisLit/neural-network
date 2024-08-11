# the layer is a wrapper for neurons
# the layer's job is to call each neuron's forward method on the inputs
# the layer returns back a list of values manipulated by the neurons

from modules.neuron import Neuron


class Layer:
    def __init__(self, shape: int, size: int):
        self.neurons = [Neuron(shape) for _ in range(size)]

    def forward(self, inputs: list[float]) -> list[float]:
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.forward(inputs))
        return outputs

    def backpropagation(
        self, expected: list[float], learning_rate: float
    ) -> list[float]:
        errors = []
        for neuron, target in zip(self.neurons, expected):
            error = neuron.backpropagation(target, learning_rate)
            errors.append(error)
        return errors
