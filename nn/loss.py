import numpy as np

class Loss:
    def forward(self, y_true, y_pred):
        raise NotImplementedError

    def backward(self, y_true, y_pred):
        raise NotImplementedError

class MSE(Loss):
    """
    Mean Squared Error Loss
    Formula: L = (1/n) * sum((y_true - y_pred)^2)
    Gradient: dL/dy_pred = (2/n) * (y_pred - y_true)
    """
    def forward(self, y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))

    def backward(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / np.size(y_true)

class BinaryCrossEntropy(Loss):
    """
    Binary Cross Entropy Loss
    Formula: L = -(1/n) * sum(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
    Gradient: dL/dy_pred = -(y_true/y_pred) + ((1-y_true)/(1-y_pred))
                         = (y_pred - y_true) / (y_pred * (1 - y_pred))
    """
    def forward(self, y_true, y_pred):
        # Clip values to prevent log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backward(self, y_true, y_pred):
        # Clip values to prevent division by 0
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return (y_pred - y_true) / (y_pred * (1 - y_pred)) / np.size(y_true)
