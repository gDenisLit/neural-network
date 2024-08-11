# the loss function is the mean squared error
# the error is the difference between the predicted value and the actual value
# the loss function is the average of the squared error


def _calculate_loss(predicted: float, actual: float) -> float:
    return (predicted - actual) ** 2


def calculate_mean_loss(predicted: list[float], actual: list[float]) -> float:
    total_loss = 0
    for i in range(len(predicted)):
        total_loss += _calculate_loss(predicted[i], actual[i])
    return total_loss / len(predicted)
