import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
from math_utils import sigmoid, sigmoid_derivative, softmax, categorical_loss_entropy

class Test_Math(unittest.TestCase):
    def test_softmax(self):
        result = softmax([1, 2, 3])
        for res, expected in zip(result, [0.09003057, 0.24472847, 0.66524096]):
            self.assertAlmostEqual(res, expected)
    
    def test_categorical_entropy_loss(self):
        result = categorical_loss_entropy([0, 1, 0],[0.2, 0.7, 0.1])
        self.assertAlmostEqual(result, 0.356674943)
