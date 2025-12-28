import unittest
import numpy as np
import sys
import os

# Create relative path to nn module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nn.activations import Sigmoid, ReLU
from nn.loss import MSE, BinaryCrossEntropy
from nn.layers import Dense
from nn.network import NeuralNetwork

class TestActivations(unittest.TestCase):
    def test_sigmoid_forward(self):
        act = Sigmoid()
        res = act.forward(np.array([0]))
        self.assertAlmostEqual(res[0], 0.5)

    def test_relu_forward(self):
        act = ReLU()
        input_data = np.array([-1, 0, 1])
        res = act.forward(input_data)
        np.testing.assert_array_equal(res, np.array([0, 0, 1]))

class TestLoss(unittest.TestCase):
    def test_mse_forward(self):
        loss = MSE()
        y_true = np.array([1.0])
        y_pred = np.array([0.5])
        # MSE = (1 - 0.5)^2 = 0.25
        self.assertAlmostEqual(loss.forward(y_true, y_pred), 0.25)

class TestLayer(unittest.TestCase):
    def test_dense_shapes(self):
        # 2 inputs, 3 outputs
        layer = Dense(2, 3)
        input_data = np.random.randn(1, 2)
        output = layer.forward(input_data)
        self.assertEqual(output.shape, (1, 3))

class TestIntegration(unittest.TestCase):
    def test_xor_convergence(self):
        # Mini integration test
        nn = NeuralNetwork(MSE())
        nn.add(Dense(2, 3))
        nn.add(Sigmoid())
        nn.add(Dense(3, 1))
        nn.add(Sigmoid())
        
        # Simple XOR subset
        X = np.array([[0, 0], [1, 1]])
        y = np.array([[0], [0]]) # Easy case
        
        # Train a bit
        initial_loss = nn.train_one_epoch(X, y, 0.1)
        for _ in range(100):
            final_loss = nn.train_one_epoch(X, y, 0.1)
            
        self.assertLess(final_loss, initial_loss, "Loss should decrease")

if __name__ == '__main__':
    unittest.main()
