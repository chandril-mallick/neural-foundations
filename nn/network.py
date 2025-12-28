import numpy as np

class NeuralNetwork:
    def __init__(self, loss_function):
        self.layers = []
        self.loss_function = loss_function

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, input):
        # Result list
        result = []
        
        # Handle single sample or batch
        if input.ndim == 1:
            input = input.reshape(1, -1)
            
        samples = len(input)

        # Run network over all samples
        for i in range(samples):
            # Shape to (1, features)
            output = input[i].reshape(1, -1)
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)

        return result

    def train_one_epoch(self, x_train, y_train, learning_rate):
        error = 0
        for x, y in zip(x_train, y_train):
            # Reshape to (1, features) for consistent matrix operations
            x = np.reshape(x, (1, -1))
            y = np.reshape(y, (1, -1))
            
            # Forward pass
            output = x
            for layer in self.layers:
                output = layer.forward(output)
            
            # Compute loss
            error += self.loss_function.forward(y, output)
            
            # Backward pass
            grad = self.loss_function.backward(y, output)
            
            for layer in reversed(self.layers):
                grad = layer.backward(grad, learning_rate)
        
        return error / len(x_train)

    def train(self, x_train, y_train, epochs, learning_rate, verbose=True):
        loss_history = []
        
        for epoch in range(epochs):
            error = self.train_one_epoch(x_train, y_train, learning_rate)
            loss_history.append(error)
            
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}, error={error:.6f}")
                
        return loss_history
