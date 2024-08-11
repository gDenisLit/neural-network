# each neuron has its own weights and bias
# the neuron job is to pass information forward and backwards
# when passing information forward, the neuron's job is to influence the passed value
# when passing information backwards, the neuron's job is to update the weight and bias
# forward propagation is what creates the "function"
# backpropagation is what causes the "learning", by learning we mean holding a certain state for wights and bias

# the neuron should be able to create its own weights and bias
# it will define them randomly
# the amount of weights should be equal to the amount of inputs (the size/shape of the previous layer)

import random
from functions.relu import relu, relu_derivative


class Neuron:
    def __init__(self, size: int):
        self.weights: list[float] = []
        self.bias: float = 0.1
        self._define_weights(size)
        self._define_bias()
        self.activated_output: float = 0
        self.output: float = 0
        self.inputs: list[float] = []

    def forward(self, inputs: list[float]) -> float:
        self.output = 0
        self.inputs = inputs

        for i in range(len(inputs)):
            self.output += inputs[i] * self.weights[i]

        self.output += self.bias
        self.activated_output = relu(self.output)

        return self.activated_output

    def _define_weights(self, size: int):
        self.weights = [random.random() for _ in range(size)]

    def _define_bias(self):
        self.bias = random.random()

    def backpropagation(self, target: float, learning_rate: float) -> float:
        loss_gradient = self.activated_output - target
        output_derivative = relu_derivative(self.output)

        for i in range(len(self.weights)):
            weight_gradient = loss_gradient * output_derivative * self.inputs[i]
            self.weights[i] -= learning_rate * weight_gradient

        bias_gradient = loss_gradient * output_derivative
        self.bias -= learning_rate * bias_gradient

        return loss_gradient * output_derivative
