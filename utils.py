import numpy as np

def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]

def generate_xor_data():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # Reshape for cleaner matrix operations depending on implementation, 
    # but here we keep as (N, 2).
    # Output needs to be (N, 1) for Binary Cross Entropy or MSE
    y = np.array([[0], [1], [1], [0]])
    return X, y

def generate_blobs(samples=100):
    # Simple 2-class blobs
    np.random.seed(42)
    # Class 0: centered at (0.2, 0.2)
    X0 = np.random.randn(samples // 2, 2) * 0.1 + np.array([0.2, 0.2])
    # Class 1: centered at (0.8, 0.8)
    X1 = np.random.randn(samples // 2, 2) * 0.1 + np.array([0.8, 0.8])
    
    X = np.vstack((X0, X1))
    y = np.vstack((np.zeros((samples // 2, 1)), np.ones((samples // 2, 1))))
    
    # Shuffle
    indices = np.arange(samples)
    np.random.shuffle(indices)
    return X[indices], y[indices]
