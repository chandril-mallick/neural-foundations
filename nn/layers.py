import numpy as np

class Layer:
    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_gradient, learning_rate):
        raise NotImplementedError

class Dense(Layer):
    def __init__(self, input_size, output_size):
        # Weight Initialization: He Initialization
        # Good for ReLU. Scales weights by sqrt(2/n) to keep variance constant.
        # For Sigmoid, Xavier (sqrt(1/n)) is often used, but He is robust.
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.bias = np.zeros(output_size)
    
    def forward(self, input):
        self.input = input
        # Y = X . W + B
        return np.dot(self.input, self.weights) + self.bias

    def backward(self, output_gradient, learning_rate):
        # Gradients:
        # dE/dW = X.T . dE/dY
        # dE/dB = sum(dE/dY)
        # dE/dX = dE/dY . W.T
        
        weights_gradient = np.dot(self.input.T, output_gradient)
        bias_gradient = np.sum(output_gradient, axis=0) # Sum over batch
        input_gradient = np.dot(output_gradient, self.weights.T)

        # Update parameters (Gradient Descent)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient

        return input_gradient
