import numpy as np
from nn.network import NeuralNetwork
from nn.layers import Dense
from nn.activations import Sigmoid
from nn.loss import MSE
from utils import generate_xor_data

def test_xor():
    # Load Data
    X, y = generate_xor_data()
    
    # Re-build and train (or load weights if we had saving)
    # For this simple project, we train a fresh model to ensure verifying the *code* works.
    nn = NeuralNetwork(loss_function=MSE())
    nn.add(Dense(2, 3))
    nn.add(Sigmoid())
    nn.add(Dense(3, 1))
    nn.add(Sigmoid())
    
    print("Training for testing...")
    nn.train(X, y, epochs=5000, learning_rate=0.1, verbose=False)
    
    predictions = nn.predict(X)
    print("\nTest Results:")
    for i, (input, target) in enumerate(zip(X, y)):
        pred = predictions[i][0,0]
        print(f"Input: {input}, Target: {target[0]}, Pred: {pred:.4f}")
        
    # Simple assertion for a passed test
    # (Checking if predictions are on correct side of 0.5)
    binary_preds = [1 if p[0,0] > 0.5 else 0 for p in predictions]
    expected = [0, 1, 1, 0]
    
    if binary_preds == expected:
        print("\n[PASS] XOR Test Passed!")
    else:
        print("\n[FAIL] XOR Test Failed.")

if __name__ == "__main__":
    test_xor()
