# The rectified linear unit (ReLU) activation function introduces the property of nonlinearity
# to a deep learning model and solves the vanishing gradients issue


def relu(x: float) -> float:
    return max(0, x)


def relu_derivative(x: float) -> float:
    return 1.0 if x > 0 else 0.0
