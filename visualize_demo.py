import numpy as np
import matplotlib.pyplot as plt
from nn.network import NeuralNetwork
from nn.layers import Dense
from nn.activations import Sigmoid
from nn.loss import MSE
from utils import generate_xor_data

def track_training():
    X, y = generate_xor_data()
    
    # Setup Network manually to access internals
    dense1 = Dense(2, 3)
    act1 = Sigmoid()
    dense2 = Dense(3, 1)
    act2 = Sigmoid()
    
    loss_fn = MSE()
    
    epochs = 10000
    learning_rate = 0.1
    
    loss_history = []
    # Track weights from input -> hidden layer (2x3 matrix)
    # We will track 3 specific weights: w[0,0], w[0,1], w[1,0]
    weight_history = {
        'w1_00': [], 'w1_01': [], 'w1_10': [], 
        'w2_00': [] # One weight from hidden->output
    }
    
    print("Training and tracking dynamics...")
    
    for epoch in range(epochs):
        error = 0
        for x_i, y_i in zip(X, y):
            x_i = x_i.reshape(1, -1)
            y_i = y_i.reshape(1, -1)
            
            # Forward
            h1 = dense1.forward(x_i)
            a1 = act1.forward(h1)
            h2 = dense2.forward(a1)
            out = act2.forward(h2)
            
            error += loss_fn.forward(y_i, out)
            
            # Backward
            grad = loss_fn.backward(y_i, out)
            grad = act2.backward(grad, learning_rate)
            grad = dense2.backward(grad, learning_rate)
            grad = act1.backward(grad, learning_rate)
            grad = dense1.backward(grad, learning_rate)
        
        loss_history.append(error / len(X))
        
        # Log weights
        weight_history['w1_00'].append(dense1.weights[0, 0])
        weight_history['w1_01'].append(dense1.weights[0, 1])
        weight_history['w1_10'].append(dense1.weights[1, 0])
        weight_history['w2_00'].append(dense2.weights[0, 0])

    return dense1, act1, dense2, act2, loss_history, weight_history

def plot_decision_boundary(d1, a1, d2, a2):
    print("Generating decision boundary...")
    # Create meshgrid
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict for all points
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = []
    for point in grid:
        point = point.reshape(1, -1)
        out = a2.forward(d2.forward(a1.forward(d1.forward(point))))
        Z.append(out[0,0])
    
    Z = np.array(Z).reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.colorbar(label='Prediction Confidence')
    
    # Plot data points
    X, y = generate_xor_data()
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), s=200, cmap=plt.cm.Spectral, edgecolors='k')
    plt.title('XOR Decision Boundary')
    plt.savefig('assets/decision_boundary.png')
    plt.close()

def plot_weight_dynamics(history):
    print("Generating weight dynamics plot...")
    plt.figure(figsize=(10, 6))
    for key, val in history.items():
        plt.plot(val, label=key)
    plt.title('Weight Dynamics during Training')
    plt.xlabel('Epochs')
    plt.ylabel('Weight Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('assets/weight_dynamics.png')
    plt.close()

def plot_hidden_activations(d1, a1):
    print("Generating hidden layer activations...")
    # Meshgrid again
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Computer activations for all points
    activations = []
    for point in grid:
        point = point.reshape(1, -1)
        z = d1.forward(point)
        a = a1.forward(z) # Shape (1, 3)
        activations.append(a[0])
    
    activations = np.array(activations) # (N, 3)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        Z = activations[:, i].reshape(xx.shape)
        ax = axes[i]
        c = ax.contourf(xx, yy, Z, cmap='viridis')
        ax.set_title(f'Neuron {i+1} Activation')
        ax.scatter([0, 0, 1, 1], [0, 1, 0, 1], c='red', s=50, edgecolors='k') # Reference points
        fig.colorbar(c, ax=ax)
    
    plt.suptitle('Learned Representations for Hidden Neurons')
    plt.savefig('assets/hidden_activations.png')
    plt.close()

def plot_loss(loss):
    plt.figure(figsize=(8,5))
    plt.plot(loss)
    plt.title("Convergence (MSE Loss)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig('assets/training_loss.png')
    plt.close()

if __name__ == "__main__":
    d1, a1, d2, a2, loss, weights = track_training()
    
    plot_decision_boundary(d1, a1, d2, a2)
    plot_weight_dynamics(weights)
    plot_hidden_activations(d1, a1)
    plot_loss(loss)
    print("All visualizations saved to assets/")
