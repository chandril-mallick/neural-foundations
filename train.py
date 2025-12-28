import numpy as np
import matplotlib.pyplot as plt

from nn.network import NeuralNetwork
from nn.layers import Dense
from nn.activations import Sigmoid, ReLU
from nn.loss import MSE, BinaryCrossEntropy
from utils import generate_xor_data

def train_xor():
    # 1. Generate Data
    X, y = generate_xor_data()
    
    # 2. Create Network
    # Architecture: 2 inputs -> 3 hidden neurons (ReLU) -> 1 output neuron (Sigmoid)
    nn = NeuralNetwork(loss_function=MSE())
    nn.add(Dense(2, 3))
    nn.add(Sigmoid()) # Using Sigmoid in hidden layer often works easier for XOR with simple SGD
    nn.add(Dense(3, 1))
    nn.add(Sigmoid())
    
    # 3. Train
    print("Training XOR...")
    loss_history = nn.train(X, y, epochs=10000, learning_rate=0.1, verbose=True)
    
    # 4. Test
    print("\npredictions:")
    predictions = nn.predict(X)
    for x_sample, pred in zip(X, predictions):
        print(f"Input: {x_sample}, Pred: {pred[0,0]:.4f}")
        
    # 5. Plot Loss
    plt.plot(loss_history)
    plt.title('Training Loss (XOR)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('xor_training_loss.png')
    print("\nLoss plot saved to xor_training_loss.png")

if __name__ == "__main__":
    train_xor()
