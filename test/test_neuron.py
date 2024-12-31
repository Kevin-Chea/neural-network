import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
from neuron import Neuron
from math_utils import sigmoid

class Test_Neuron(unittest.TestCase):
    def test_activate(self):
        self.assertEqual(1, 1)
        neuron = Neuron(2)
        neuron.weights = [0.1, 0.5]
        neuron.bias = 1
        inputs = [1, 2]
        output = neuron.activate(inputs)
        # sigmoid(1 * 0.1 + 2 * 0.5 + 1) = sigmoid(2.1)
        self.assertEqual(neuron.z, 2.1)
        self.assertEqual(output, sigmoid(2.1))
    
    def test_update_weights(self):
        learning_rate = 0.1
        neuron = Neuron(2)
        neuron.weights = [0.1, 0.5]
        neuron.weight_gradients = [-1, 2]
        neuron.bias = 1
        neuron.bias_gradient = -1
        neuron.update_weights(learning_rate)
        self.assertListEqual(neuron.weights, [0.2, 0.3])
        self.assertEqual(neuron.bias, 1.1)
    
    def test_compute_gradient(self):
        neuron = Neuron(2)
        neuron.z = 3
        neuron.inputs = [0, 1]
        neuron.compute_gradient(0.5)
        
        expected_almost_delta = 0.04517665973086899 / 2
        self.assertAlmostEqual(neuron.delta, expected_almost_delta)
        self.assertAlmostEqual(neuron.bias_gradient, expected_almost_delta)
        self.assertListEqual(neuron.weight_gradients, [0, neuron.delta])
        

if __name__ == '__main__':
    unittest.main()